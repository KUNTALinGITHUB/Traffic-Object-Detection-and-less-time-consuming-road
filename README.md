Traffic Object Detection and less time consuming read

This project uses the YOLO model to detect and analyze traffic objects in images. The application is built using Streamlit, OpenCV, and Matplotlib.

Features
Object Detection: Detects objects such as bicycles, buses, cars, motorbikes, and persons in uploaded images.
Annotated Images: Saves and displays images with detected objects annotated.
Traffic Analysis: Counts the number of each type of object and provides a summary.
Bar Graph: Generates a bar graph showing the total number of objects detected in each image.
Least Traffic Detection: Identifies and highlights the image with the least traffic.
Installation
Clone the repository:


Install the required packages:



Download the YOLO model weights:

Place the best.pt file in the appropriate directory as specified in the code.
Usage
Run the Streamlit app:


Upload Images:

Use the file uploader in the Streamlit app to upload images for object detection.
View Results:

The app will display annotated images, a bar graph of total objects detected, and highlight the image with the least traffic.
Code Overview
app.py: Main application file containing the Streamlit app.
process_image: Function to process a single image, detect objects, and save annotated images and counts.
process_folder: Function to process all images in a folder, generate a bar graph, and identify the image with the least traffic.
clear_output_folder: Function to clear the output folder before processing new images.
Example
!Bar Graph

Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

License
This project is licensed under the MIT License. See the LICENSE file for details.

