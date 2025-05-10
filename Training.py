from ultralytics import YOLO

charclassnames = [ "0", 
             "1", "2", "3", "4", "5",
             "6", "7", "8", "9", "b",
             "d", "h", "v", "t", "ta",
             "y", "n", "s", "sad", "l",
             "j", "m", "g", "e", "wh"]

# Load pre-trained YOLOv8 model (smallest for OCR task)
model = YOLO("yolo11s.pt")

# Train the model on your OCR dataset
model.train(
    data="data.yaml",  # Define dataset YAML (train/val paths)
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size
    batch=8,  # Adjust per GPU memory
    project="OCR_YOLO",  # Save directory
    name="yolov11_ocr",  # Experiment name
    device="cuda"  # Use GPU if available
)

# # Evaluate model performance on the validation set
# metrics = model.val()

# # Perform object detection on an image
# results = model("path/to/image.jpg")
# results[0].show()
