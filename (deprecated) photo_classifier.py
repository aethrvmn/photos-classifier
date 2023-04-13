import os
import urllib.request
from PIL import Image
from torchvision import transforms
import torch
from tqdm import tqdm
import ssl
import numpy as np
from skimage.metrics import structural_similarity as ssim

if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

# Load the pre-trained model
model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl')
model.eval()

# Load ImageNet labels
labels_url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
categories = [s.strip().decode('utf-8') for s in urllib.request.urlopen(labels_url).readlines()]

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

custom_mapping = {
    # Add your custom mappings here
}

def get_image_files_recursively(input_folder):
    image_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')) and not file.startswith('.'):
                image_files.append(os.path.join(root, file))
    return image_files

def classify_image(image_path, custom_mapping=None):
    input_image = Image.open(image_path)
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top1_prob, top1_catid = torch.topk(probabilities, 1)

    if custom_mapping:
        return custom_mapping[categories[top1_catid[0]]], top1_prob[0].item()
    else:
        return categories[top1_catid[0]], top1_prob[0].item()

def calculate_image_similarity(img1, img2):
    img1_preprocessed = preprocess(img1).mean(dim=0).numpy()
    img2_preprocessed = preprocess(img2).mean(dim=0).numpy()

    return ssim(img1_preprocessed, img2_preprocessed, data_range=img1_preprocessed.max() - img1_preprocessed.min())




def is_duplicate(image, output_folder, label, threshold=0.95):
    dest_folder = os.path.join(output_folder, label)

    if not os.path.exists(dest_folder):
        return False

    for existing_image in os.listdir(dest_folder):
        existing_image_path = os.path.join(dest_folder, existing_image)
        with Image.open(existing_image_path) as existing_img:
            similarity = calculate_image_similarity(image, existing_img)
            if similarity >= threshold:
                return True

    return False

def get_next_filename(dest_folder, label, ext):
    i = 1
    while True:
        new_filename = f"{label}{i}.{ext}"
        new_filepath = os.path.join(dest_folder, new_filename)
        if not os.path.exists(new_filepath):
            return new_filename
        i += 1

input_folder = 'photos/input'
output_folder = 'photos/output'

image_files = get_image_files_recursively(input_folder)

for image_path in tqdm(image_files):
    label, _ = classify_image(image_path)
    dest_folder = os.path.join(output_folder, label)

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    with Image.open(image_path) as input_image:
        if is_duplicate(input_image, output_folder, label):
            print(f"Skipping duplicate: {image_path}")
            continue

        ext = os.path.splitext(image_path)[1].lower().lstrip('.')
        new_filename = get_next_filename(dest_folder, label, ext)
        dest_path = os.path.join(dest_folder, new_filename)
        os.rename(image_path, dest_path)
        print(f'Moved {image_path} to {dest_path}')
