import pandas as pd
import cv2
import numpy as np
import os
from datetime import datetime
from PIL import Image
from collections import Counter
import shutil
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from torchvision.models import resnet50, ResNet50_Weights
net = cv2.dnn.readNet('frozen_east_text_detection.pb')
def detect_and_crop_text(image_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    orig = image.copy()
    (H, W) = image.shape[:2]

    (newW, newH) = (640, 640)
    rW = W / float(newW)
    rH = H / float(newH)

    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)

    net.setInput(blob)

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",  # Для вероятности наличия текста
        "feature_fusion/concat_3"        # Для координат рамок текста
    ]

    (scores, geometry) = net.forward(layerNames)

    (rects, confidences) = decode_predictions(scores, geometry)

    if len(rects) == 0:
        no=0
        print("No text blocks detected.")
        return no

    indices = cv2.dnn.NMSBoxes(rects, confidences, 0.5, 0.4)

    if len(indices) > 0:
        indices = indices.flatten()

        for i in indices:
            (startX, startY, endX, endY) = rects[i]

            if startX < 0 or startY < 0 or endX > W or endY > H:
                print(f"Invalid coordinates: ({startX}, {startY}, {endX}, {endY})")
                continue

            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            cropped_text_block = orig[startY:endY, startX:endX]

            if cropped_text_block.size == 0:
                print(f"Empty cropped text block: ({startX}, {startY}, {endX}, {endY})")
                continue

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')
            output_path = os.path.join(output_dir, f'text_block_{timestamp}_{i}.png')
            cv2.imwrite(output_path, cropped_text_block)
            #cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
    #cv2_imshow(orig)

def decode_predictions(scores, geometry):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < 0.5:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_features = 2048
out_features = 4
model = resnet50(weights=None)
model.fc = nn.Linear(in_features, out_features)

model.load_state_dict(torch.load('resnet_4.pth', map_location=device))
model = model.to(device)
model.eval()

transformer = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB") if img.mode != 'RGB' else img),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def classify_text_block(image_path):

    image = Image.open(image_path).convert('RGB')
    with torch.no_grad():
        input_tensor = transformer(image).unsqueeze(0)
        input_tensor = input_tensor.to(device)
        model.eval()
        output = model(input_tensor)
        predicted_class = torch.argmax(output, 1).item()

    class_mapping = {0: '270deg', 1: '180deg', 2: '90deg', 3: '0deg'}
    predicted_orientation = class_mapping[predicted_class]
    #print(predicted_orientation)
    return predicted_orientation

def classify_image(model, image_path, output_dir):
    detect_and_crop_text(image_path, output_dir)
    orientations = []
    files = [f for f in os.listdir(output_dir) if f.endswith(".jpg") or f.endswith(".png")]

    if len(files) > 0:

        if len(files) > 20:
            random_files = random.sample(files, 20)
        else:
            random_files = files
        #random_files = files
        for filename in random_files:
            image_path = os.path.join(output_dir, filename)
            orientation = classify_text_block(image_path)
            orientations.append(orientation)

        most_common_orientation = max(set(orientations), key=orientations.count)
        shutil.rmtree(output_dir)
    else:
        most_common_orientation = 'Пожалуйста, загрузите картинку в лучшем разрешении'

    return most_common_orientation
def predict_orientation(image_path):
    output_dir = 'temp'
    im = Image.open(image_path)
    predicted_orientation = classify_image(model, image_path, output_dir)
    print(predicted_orientation)
    if predicted_orientation == '0deg':
        image_path3 = image_path
    if predicted_orientation == '180deg':
        out = im.rotate(-180, expand=True)
        out.save('out.png')
        image_path3 = 'out.png'
    if predicted_orientation == 'Пожалуйста, загрузите картинку в лучшем разрешении':
        image_path3 = 'Ошибка'
    if (predicted_orientation == '270deg') or (predicted_orientation == '90deg'):
        out = im.rotate(-90, expand=True)
        out.save('1.png')
        image_path2 = '1.png'
        predicted_orientation = classify_image(model, image_path2, output_dir)
        im = Image.open(image_path2)
        if predicted_orientation == '180deg':
            out = im.rotate(-180, expand=True)
            out.save('out.png')
            image_path3 = 'out.png'
        if predicted_orientation == '0deg':
            image_path3 = image_path2
    return image_path3