import os

# Define the path to your dataset
dataset_path = "datasets/data_modified/valid/labels"
images_path = "datasets/data_modified/valid/images"
# Loop through all .txt files in the dataset directory
for label_file in os.listdir(dataset_path):
    if label_file.endswith(".txt"):
        label_path = os.path.join(dataset_path, label_file)
        
        # Check if the label file is empty
        if os.path.getsize(label_path) == 0:
            # Construct the corresponding image path (YOLO supports jpg, png, etc.)
            base_name = os.path.splitext(label_file)[0]
            image_extensions = [".jpg", ".png", ".jpeg"]

            # Delete the empty label file
            os.remove(label_path)
            print(f"Deleted empty label: {label_path}")

            # Delete the associated image if it exists
            for ext in image_extensions:
                image_path = os.path.join(images_path, base_name + ext)
                if os.path.exists(image_path):
                    os.remove(image_path)
                    print(f"Deleted image: {image_path}")
