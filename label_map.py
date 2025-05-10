label_map = {
    "0": "0",
    "1": "9",
    "2": "10",
    "3": "11",
    "4": "24",
    "5": "23",
    "6": "21",
    "7": "23",
    "8": "12",
    "9": "20",
    "10": "22",
    "11": "1",
    "12": "25",
    "13": "17",
    "14": "18",
    "15": "19",
    "16": "14",
    "17": "15",
    "18": "13",
    "19": "16",
    "20": "2",
    "21": "3",
    "22": "4",  # Example: If you want to change 21 to 1, add it here
    "23": "5",
    "24": "6",
    "25": "7",
    "26": "8"
}

# label_map = [ "0", 
#              "1", "2", "3", "4", "5",
#              "6", "7", "8", "9", "b",
#              "d", "h", "v", "t", "ta",
#              "y", "n", "s", "sad", "l",
#              "j", "m", "g", "e", "wh"]

import os
import shutil

# Define paths
labels_dir = "datasets/data/valid/labels"  # Directory containing label files
output_dir = "datasets/data_modified/valid/labels"  # Directory to save images with label "6"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(labels_dir):
    if filename.endswith(".txt"):  # Ensure it's a label file
        input_file = os.path.join(labels_dir, filename)
        output_file = os.path.join(output_dir, filename)

        with open(input_file, "r") as f:
            lines = f.readlines()

        # Update label values
        updated_lines = []
        for line in lines:
            parts = line.split()
            if parts and parts[0] in label_map:
                parts[0] = label_map[parts[0]]  # Replace label
            updated_lines.append(" ".join(parts))

        # Save back to the same file (overwrite)
        with open(output_file, "w") as f:
            f.write("\n".join(updated_lines) + "\n")

        print(f"Updated labels in: {filename}")

print("All label files updated successfully!")