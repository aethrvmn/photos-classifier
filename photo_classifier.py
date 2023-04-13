import torch
import os
import shutil
import json
from PIL import Image
from torchvision import transforms

# Load the pre-trained ResNet-18 model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

# Set the model to evaluation mode
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

custom_mapping = {
    "sea": ["beach", "seashore", "coast", "ocean", "sea"],
    "forest": ["forest", "wood", "woods", "jungle"],
    "animal": ["dog", "cat", "bird", "fish", "lion", "tiger", "elephant", "zebra"],
    "people": ["person", "man", "woman", "boy", "girl"],
    "portrait": ["face", "head", "selfie"],
    "city": ["cityscape", "skyscraper", "buildings", "street"]
}


def classify_image(image_path, custom_mapping=None):
    # Open and preprocess the image
    input_image = Image.open(image_path)
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # Perform the classification
    with torch.no_grad():
        output = model(input_batch)

    # Get probabilities using softmax
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the top predicted class label
    _, predicted_label_index = torch.max(probabilities, 0)
    predicted_label = categories[predicted_label_index.item()]

    # If custom mapping is provided, map the predicted label to a custom label
    if custom_mapping:
        for custom_label, keywords in custom_mapping.items():
            if predicted_label.lower() in keywords:
                return custom_label
        return 'others'

    return predicted_label


labels_url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
categories = [s.strip() for s in urllib.request.urlopen(labels_url).readlines()]
categories = [category.decode("utf-8") for category in categories]


input_folder = "path/to/your/images"
output_folder = "path/to/output/folder"

# Loop through all image files
for file in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file)

    # Classify the image
    label = classify_image(file_path)

    # Create a folder for the label if it doesn't exist
    label_folder = os.path.join(output_folder, label)
    os.makedirs(label_folder, exist_ok=True)

    # Move the image to the label folder
    shutil.move(file_path, os.path.join(label_folder, file))
