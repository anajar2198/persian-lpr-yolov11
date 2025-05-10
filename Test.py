from ultralytics import YOLO
import cv2 

charclassnames = [ "0", 
             "1", "2", "3", "4", "5",
             "6", "7", "8", "9", "b",
             "d", "h", "v", "t", "ta",
             "y", "n", "s", "sad", "l",
             "j", "m", "g", "e", "wh"]


# Load your model
model = YOLO('OCR_YOLO/yolov11_ocr/weights/best.pt')


metrics = model.val()
# Run inference on a test image or directory
results = model.predict(source='datasets/data_modified/test/images', save=False)

for result in results:
    # Print the path of the processed image
    print("Processed Image Path:", result.path)
    
    # Original image shape
    print("Original Image Shape:", result.orig_shape)
    
    # Check if there are bounding boxes
    if result.boxes is not None:
        # Create a copy of the original image for annotation
        annotated_image = result.orig_img.copy()

        for box in result.boxes.data:  # Iterate over detected boxes
            x1, y1, x2, y2, confidence, cls = box
            
            # Map the class index to the character name
            char_name = charclassnames[int(cls)]
            print(f"Class: {char_name}, Confidence: {confidence:.2f}, Box: ({x1}, {y1}, {x2}, {y2})")
            
            # Draw the bounding box
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Draw box
            
            # Draw a filled rectangle for the background of the text
            text_background_height = 20  # Height of the text background
            cv2.rectangle(annotated_image, (int(x1), int(y1) - text_background_height), 
                          (int(x2), int(y1)), (255, 0, 0), cv2.FILLED)  # Fill with rectangle color
            
            # Add the text on top of the filled rectangle
            cv2.putText(annotated_image, f"{char_name} {confidence:.2f}", (int(x1), int(y1) - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Add label in black color
        # Save the annotated image
        annotated_image_path = f"runs/detect/predict_test/annotated_{result.path.split('/')[-1]}"
        cv2.imwrite(annotated_image_path, annotated_image)
        print(f"Annotated image saved to: {annotated_image_path}")
    
    # Print the inference speed
    print("Inference Speed:", result.speed)
