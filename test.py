import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

# Load the YOLOv8 model
model = YOLO(r"D:\Traffic\copy\runs\detect\train\weights\best.pt")

# Define class names
class_names = ['bicycle', 'bus', 'car', 'motorbike', 'person']

# Function to process a single image
def process_image(image_path, output_folder):
    image = cv2.imread(image_path)
    results = model(image)
    annotated_image = results[0].plot()

    # Save the annotated image
    image_name = os.path.basename(image_path)
    output_image_path = os.path.join(output_folder, f"annotated_{image_name}")
    cv2.imwrite(output_image_path, annotated_image)
    print(f"Image saved to {output_image_path}")

    # Create a dictionary to count the number of each class object
    class_counts = {class_name: 0 for class_name in class_names}
    total_objects = 0

    # Count the number of each class object in the image
    for result in results:
        for detection in result.boxes:
            class_index = int(detection.cls)
            class_name = class_names[class_index]
            class_counts[class_name] += 1
            total_objects += 1

    # Create a text file with class names and their counts
    txt_file_path = os.path.join(output_folder, f"annotated_{image_name}.txt")
    with open(txt_file_path, 'w') as f:
        for class_name, count in class_counts.items():
            f.write(f"Class: {class_name}, Count: {count}\n")
        f.write(f"Total Objects: {total_objects}\n")

    print(f"Detection details saved to {txt_file_path}")

    return total_objects, image_name

# Function to process all images in a folder and create a bar graph
def process_folder(folder_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    graph_folder = os.path.join(output_folder, "graphs")
    os.makedirs(graph_folder, exist_ok=True)

    total_objects_list = []
    image_names = []

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        if os.path.isfile(image_path):
            total_objects, img_name = process_image(image_path, output_folder)
            total_objects_list.append(total_objects)
            image_names.append(img_name)

    # Create a bar graph based on total_objects for each image
    plt.figure(figsize=(10, 6))
    plt.bar(image_names, total_objects_list, color='blue')
    plt.xlabel('Image Name')
    plt.ylabel('Total Objects')
    plt.title('Total Objects Detected in Each Image')
    plt.xticks(rotation=45, ha='right')
    
    # Save the bar graph in the graph folder
    graph_path = os.path.join(graph_folder, "total_objects_bar_graph.png")
    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()
    
    print(f"Bar graph saved to {graph_path}")

# Define the input and output folders
input_folder = r"D:\Traffic\copy\Results"
output_folder = "Results/Output"

# Process all images in the input folder and create a bar graph
process_folder(input_folder, output_folder)
