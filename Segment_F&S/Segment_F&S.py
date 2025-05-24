from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_name = "D:\AI_Sugar\AI_Sugar\Image\image11.png"

image_orig = cv2.imread(image_name)
original_h, original_w = image_orig.shape[:2]

# Resize image
resized = cv2.resize(image_orig, (640, 640))

model = YOLO("yolov8n-seg.pt")
results = model(resized)[0]

target_classes = ["spoon", "fork"]
masks_and_labels = []

for mask, cls_idx in zip(results.masks.data, results.boxes.cls):
    class_name = model.names[int(cls_idx)]
    if class_name in target_classes:
        masks_and_labels.append((mask.cpu().numpy(), class_name))

if not masks_and_labels:
    print("No spoon or fork detected.")
else:
    largest_mask, label = max(masks_and_labels, key=lambda m: np.sum(m[0]))

    binary_mask = (largest_mask > 0.5).astype(np.uint8) * 255
    binary_mask_resized = cv2.resize(binary_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    masked_image = cv2.bitwise_and(image_orig, image_orig, mask=binary_mask_resized)

    contours, _ = cv2.findContours(binary_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_with_contours = image_orig.copy()

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        aspect_ratio = h / w
        area = cv2.contourArea(largest_contour)

        if label == "spoon":
            if aspect_ratio > 2.8 and area > 3000:
                label_type = "ช้อนกลาง"
            else:
                label_type = "ช้อน"
        else:
            label_type = label

        cv2.rectangle(image_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image_with_contours, label_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # หา centroid ของ mask
        moments = cv2.moments(binary_mask_resized)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2  # fallback

        # หาค่าทิศทางของวัตถุ
        [vx, vy, x0, y0] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
        direction = np.array([vx, vy]).flatten()
        point_on_line = np.array([x0, y0]).flatten()

        # กรอบสี่เหลี่ยม
        xmin, ymin, xmax, ymax = x, y, x + w, y + h

        def clip_line_to_box(p, d, xmin, xmax, ymin, ymax):
            """ตัดเส้นให้พอดีกับกรอบ"""
            points = []
            eps = 1e-6

            # สร้างเส้นแบบ parametric: x = p[0] + t*d[0], y = p[1] + t*d[1]

            # เช็คจุดตัดแต่ละด้าน
            for edge_x in [xmin, xmax]:
                if abs(d[0]) > eps:
                    t = (edge_x - p[0]) / d[0]
                    y = p[1] + t * d[1]
                    if ymin <= y <= ymax:
                        points.append((int(edge_x), int(y)))

            for edge_y in [ymin, ymax]:
                if abs(d[1]) > eps:
                    t = (edge_y - p[1]) / d[1]
                    x = p[0] + t * d[0]
                    if xmin <= x <= xmax:
                        points.append((int(x), int(edge_y)))

            # เอาแค่ 2 จุดเท่านั้น
            if len(points) >= 2:
                return points[:2]
            else:
                return None

        # หาจุดตัด
        clipped_points = clip_line_to_box(point_on_line, direction, xmin, xmax, ymin, ymax)

        if clipped_points and len(clipped_points) == 2:
            pt1, pt2 = clipped_points
            cv2.line(image_with_contours, pt1, pt2, (0, 0, 255), 2)

            # คำนวณความยาว
            pixel_length = np.linalg.norm(np.array(pt2) - np.array(pt1))


        if label_type == "ช้อน":
            real_length_cm = 15.0
        elif label_type == "ช้อนกลาง":
            real_length_cm = 22.0
        elif label == "fork":
            real_length_cm = 18.0
        else:
            real_length_cm = 0.0

        if real_length_cm > 0:
            cm_per_pixel = real_length_cm / pixel_length
            text_info = f"{label_type}\nLength: {pixel_length:.2f}px\n1px ≈ {cm_per_pixel:.3f} cm"
            print(f"Label: {label_type}")
            print(f"Length in pixels: {pixel_length:.2f}")
            print(f"Estimated 1 pixel = {cm_per_pixel:.4f} cm")
        else:
            text_info = "Unknown object"
            print("Unknown object for real-world length estimation.")

        # วาดข้อความ
        text_lines = text_info.split("\n")
        text_x, text_y = x + w + 10, y + 20
        for i, line in enumerate(text_lines):
            cv2.putText(image_with_contours, line, (text_x, text_y + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        plt.figure(figsize=(12, 6))
        plt.title("Detected Spoon/Fork with Info")
        plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
    else:
        print("No contours found.")
