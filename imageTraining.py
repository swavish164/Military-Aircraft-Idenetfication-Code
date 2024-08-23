import torch
from ultralytics import YOLO
import requests
from PIL import Image,ImageDraw
from io import BytesIO
import matplotlib.pyplot as plt


#model = YOLO("yolov8n.yaml")  # build a new model from YAML

# Train the model
#results = model.train(data="Datasets\data.yaml", epochs=100, imgsz=640)

#model = YOLO('runs\detect\\train3\weights\\best.pt')
#metrics = model.val()


#from roboflow import Roboflow
#rf = Roboflow(api_key="0dobNtJKLIT2Hwkk90pa")
#project = rf.workspace("aircraft-4uwg3").project("plane-aycuf")
#version = project.version(2)
#dataset = version.download("yolov8")
                
                
from inference_sdk import InferenceHTTPClient

# create an inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="0dobNtJKLIT2Hwkk90pa"
)
image_path = "original.jpg"
# run inference on a local image
response = CLIENT.infer(image_path, model_id="plane-aycuf/2")
img = Image.open(image_path)
draw = ImageDraw.Draw(img)

for prediction in response['predictions']:
    x_center = prediction['x']
    y_center = prediction['y']
    box_width = prediction['width']
    box_height = prediction['height']
    
    # Calculate the coordinates of the top-left and bottom-right corners
    x0 = x_center - box_width / 2
    y0 = y_center - box_height / 2
    x1 = x_center + box_width / 2
    y1 = y_center + box_height / 2
    
    # Draw the rectangle
    draw.rectangle([x0, y0, x1, y1], outline="red", width=5)
    
    # Optionally, add a label with the class name
    draw.text((x0, y0), prediction['class'], fill="red")

# Display the image
plt.imshow(img)
plt.axis('off')  # Hide axes
plt.show()