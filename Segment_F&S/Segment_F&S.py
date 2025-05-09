from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_name = "D:\AI_Sugar\AI_Sugar\Image\\add3.jpeg"

image_orig = cv2.imread(image_name)
original_h, original_w = image_orig.shape[:2]

# ลดขนาดภาพ
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
    # หาค่า mask ที่มีขนาดใหญ่สุด
    largest_mask, label = max(masks_and_labels, key=lambda m: np.sum(m[0]))

    binary_mask = (largest_mask > 0.5).astype(np.uint8) * 255

    # Resize mask กลับไปขนาดของภาพต้นฉบับ
    binary_mask_resized = cv2.resize(binary_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    # Apply mask กับภาพต้นฉบับ
    masked_image = cv2.bitwise_and(image_orig, image_orig, mask=binary_mask_resized)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Largest Mask ({label})")
    plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
