import streamlit as st
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import shutil

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

    return annotated_image, total_objects, image_name

# Function to process all images in a folder and create a bar graph
def process_folder(folder_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    graph_folder = os.path.join(output_folder, "graphs")
    os.makedirs(graph_folder, exist_ok=True)

    total_objects_list = []
    image_names = []
    annotated_images = []
    min_traffic_images = []
    min_traffic_count = float('inf')

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        if os.path.isfile(image_path):
            annotated_image, total_objects, img_name = process_image(image_path, output_folder)
            total_objects_list.append(total_objects)
            image_names.append(img_name)
            annotated_images.append((annotated_image, img_name, total_objects))

            # Update the list of images with the least traffic
            if total_objects < min_traffic_count:
                min_traffic_count = total_objects
                min_traffic_images = [img_name]
            elif total_objects == min_traffic_count:
                min_traffic_images.append(img_name)

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
    
    st.image(graph_path, caption="Total Objects Detected in Each Image")

    return annotated_images, min_traffic_images

# Function to clear the output folder
def clear_output_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)

# Streamlit app
def main():
    st.title("Traffic Object Detection")
    st.write("Upload images to detect objects and analyze traffic.")

    output_folder = "Results/Output"
    clear_output_folder(output_folder)  # Clear the output folder at the start

    uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
    if uploaded_files:
        for uploaded_file in uploaded_files:
            image_path = os.path.join(output_folder, uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        annotated_images, min_traffic_images = process_folder(output_folder, output_folder)
        
        # Display annotated images two in a row
        for i in range(0, len(annotated_images), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(annotated_images):
                    annotated_image, img_name, total_objects = annotated_images[i + j]
                    cols[j].image(annotated_image, caption=f"Annotated {img_name}, Total Objects: {total_objects}", use_column_width=True)

        st.markdown(f"<span style='color:green; font-weight:bold;'>This is the less traffic road: {', '.join(min_traffic_images)}</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
