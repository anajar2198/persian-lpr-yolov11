from ultralytics import YOLO
import cv2
import os
import shutil
# Class labels
charclassnames = ["0", "1", "2", "3", "4", "5",
                  "6", "7", "8", "9", "b", "d",
                  "h", "v", "t", "ta", "y", "n",
                  "s", "sad", "l", "j", "m", "g",
                  "e", "wh"]

# Paths
image_dir = "datasets/data_modified/train/images"
label_dir = "datasets/data_modified/train/labels"
output_dir = "runs/detect/wrong"
os.makedirs(output_dir, exist_ok=True)

# Load YOLO model
model = YOLO("OCR_YOLO/yolov11_ocr/weights/best.pt")

def load_ground_truth_boxes(label_path, img_width, img_height):
    """
    Loads bounding boxes and labels from annotation files.
    Assumes annotations are in YOLO format (normalized): 
      class x_center y_center width height
    and converts them to pixel coordinates.
    """
    gt_boxes = []  # List of (x1, y1, x2, y2, label)
    if os.path.exists(label_path):
        with open(label_path, "r") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_index = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # Convert YOLO normalized format to pixel coordinates
                    x1 = (x_center - width / 2) * img_width
                    y1 = (y_center - height / 2) * img_height
                    x2 = (x_center + width / 2) * img_width
                    y2 = (y_center + height / 2) * img_height
                    
                    gt_boxes.append((x1, y1, x2, y2, charclassnames[class_index]))
    return gt_boxes

def yolo_to_pixel(box, img_width, img_height):
    """
    Convert YOLO format (normalized x_center, y_center, width, height) 
    to pixel coordinates (x1, y1, x2, y2).
    """
    x_center, y_center, width, height = box
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    x1 = max(0, x_center - width / 2)
    y1 = max(0, y_center - height / 2)
    x2 = min(img_width, x_center + width / 2)
    y2 = min(img_height, y_center + height / 2)
    return (x1, y1, x2, y2)

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Both boxes are assumed to be in pixel coordinates: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    # Calculate intersection
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height

    if intersection == 0:
        return 0.0  # No overlap

    # Calculate union
    box_area = (x2 - x1) * (y2 - y1)
    gt_box_area = (x2g - x1g) * (y2g - y1g)
    union = box_area + gt_box_area - intersection

    return intersection / union if union > 0 else 0.0

# Run inference
results = model.predict(source=image_dir, save=False)

for result in results:
    image_name = os.path.basename(result.path)
    
    # Get image dimensions from the original image
    img_height, img_width, _ = result.orig_img.shape
    
    # Load ground truth boxes (converted to pixel coordinates)
    label_path = os.path.join(label_dir, image_name.replace(".jpg", ".txt"))
    ground_truth_boxes = load_ground_truth_boxes(label_path, img_width, img_height)

    # Copy original image
    annotated_image = result.orig_img.copy()
    incorrect_detection = False  # Flag if any incorrect predictions exist

    if result.boxes is not None:
        for box in result.boxes.data:
            # Convert predicted box from tensor to list of scalars
            x1, y1, x2, y2, confidence, cls = box.tolist()
            predicted_label = charclassnames[int(cls)]

            best_iou = 0
            matched_gt_label = None
            
            # Compare predicted box against each ground truth box
            for (gt_x1, gt_y1, gt_x2, gt_y2, gt_label) in ground_truth_boxes:
                iou = calculate_iou((x1, y1, x2, y2), (gt_x1, gt_y1, gt_x2, gt_y2))
                if iou > best_iou:
                    best_iou = iou
                    matched_gt_label = gt_label

            # Consider a correct prediction if IoU > 0.5 and labels match
            if best_iou > 0.5 and predicted_label == matched_gt_label:
                color = (255, 0, 0)  # Blue for correct predictions
            else:
                color = (0, 0, 255)  # Red for incorrect predictions
                incorrect_detection = True

                # Draw the bounding box
                cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # Draw a filled rectangle for text background
                cv2.rectangle(annotated_image, (int(x1), int(y1) - 20), (int(x2), int(y1)), color, cv2.FILLED)

                # Add text label
                cv2.putText(annotated_image, f"P: {predicted_label} R: {matched_gt_label}",
                            (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Save image only if there was an incorrect detection
    if incorrect_detection:
        annotated_image_path = os.path.join(output_dir, f"wrong_{image_name}")
        cv2.imwrite(annotated_image_path, annotated_image)
        print(f"Annotated image saved with incorrect predictions: {annotated_image_path}")
        
        # Copy the original image file to the output folder
        origin_copy_path = os.path.join(output_dir, f"{image_name}")
        shutil.copy(result.path, origin_copy_path)
        print(f"Original image copied to: {origin_copy_path}")
        
        # Copy the label file, if it exists
        if os.path.exists(label_path):
            label_copy_path = os.path.join(output_dir, f"{os.path.basename(label_path)}")
            shutil.copy(label_path, label_copy_path)
            print(f"Label file copied to: {label_copy_path}")

    print(f"Processed Image: {image_name}, Inference Speed: {result.speed}")
