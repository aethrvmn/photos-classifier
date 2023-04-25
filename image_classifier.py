import os
import urllib.request
import shutil
import random
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
import torch
import ssl
from skimage.metrics import structural_similarity as ssim
import timm
from tqdm import tqdm

class ImageClassifier:

    def __init__(self, ssl_unverified=False, custom_mapping=None):

        if ssl_unverified and hasattr(ssl, '_create_unverified_context'):
            ssl._create_default_https_context = ssl._create_unverified_context

        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model.eval()
        self.labels_url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
        self.categories = [s.strip().decode('utf-8') for s in urllib.request.urlopen(self.labels_url).readlines()]
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.custom_mapping = custom_mapping if custom_mapping else {}

    def get_image_files_recursively(self, input_folder):
        image_files = [os.path.join(root, file) for root, _, files in os.walk(input_folder) for file in files if
                    file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')) and not file.startswith('.')]

        random.shuffle(image_files)
        return image_files

    def convert_to_rgb(self, image):
        return image.convert("RGB") if image.mode != "RGB" else image

    def classify_image(self, image_path, input_image=None):
        input_image = input_image or Image.open(image_path)
        input_image = self.convert_to_rgb(input_image)
        input_batch = self.preprocess(input_image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top1_prob, top1_catid = torch.topk(probabilities, 1)
        category = self.custom_mapping.get(self.categories[top1_catid[0]], self.categories[top1_catid[0]])
        return category, top1_prob.item()

    def calculate_image_similarity(self, img1, img2):
        img1_preprocessed = self.preprocess(self.convert_to_rgb(img1)).mean(dim=0).numpy()
        img2_preprocessed = self.preprocess(self.convert_to_rgb(img2)).mean(dim=0).numpy()
        return ssim(img1_preprocessed, img2_preprocessed, data_range=img1_preprocessed.max() - img1_preprocessed.min())

    def is_duplicate(self, image, output_folder, label, threshold=0.95):
        dest_folder = os.path.join(output_folder, label)
        if not os.path.exists(dest_folder):
            return False
        for existing_file in os.listdir(dest_folder):
            if existing_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')) and not existing_file.startswith('.'):
                with Image.open(os.path.join(dest_folder, existing_file)) as existing_img:
                    if self.calculate_image_similarity(self.convert_to_rgb(image), self.convert_to_rgb(existing_img)) >= threshold:
                        return True
        return False

    def get_next_filename(self, folder, prefix, ext):
        i = 1
        while True:
            filename_underscore = f"{prefix}_{i}.{ext}"
            filename_no_underscore = f"{prefix}{i}.{ext}"
            if not os.path.exists(os.path.join(folder, filename_underscore)) and not os.path.exists(os.path.join(folder, filename_no_underscore)):
                return filename_no_underscore
            i += 1

    def convert_to_jpeg(self, image_path):
        file_ext = os.path.splitext(image_path)[1].lower()
        input_image = self.open_image_with_warning(image_path)
        if input_image is None:
            return None, None
        image_format = input_image.format.lower()

        if file_ext == '.jpg' and image_format == 'jpeg':
            return image_path, input_image
        else:
            filename = os.path.basename(image_path)
            new_image_path = os.path.join('photos', 'converted', os.path.splitext(filename)[0] + '.jpg')
            os.makedirs(os.path.dirname(new_image_path), exist_ok=True)

            self.convert_to_rgb(input_image).save(new_image_path, 'JPEG', quality=90)
            os.remove(image_path)
            return new_image_path, input_image

    def open_image_with_warning(self, image_path):
        try:
            return Image.open(image_path)
        except UnidentifiedImageError:
            print(f"Warning: Unidentified or corrupted image: {image_path}")
            return None


    def process_images(self, input_folder, output_folder, rename_files=True, check_duplicates=True, move_files=True, delete_duplicates=True, delete_empty_folders=True, convert_to_jpeg=True):
        image_files = self.get_image_files_recursively(input_folder)

        for image_path in tqdm(image_files):
            if convert_to_jpeg:
                image_path, input_image = self.convert_to_jpeg(image_path)
            else:
                input_image = self.open_image_with_warning(image_path)

            if input_image is None:
                continue

            label, _ = self.classify_image(image_path, input_image=input_image)
            dest_folder = os.path.join(output_folder, label)
            os.makedirs(dest_folder, exist_ok=True)

            if check_duplicates and self.is_duplicate(input_image, output_folder, label):
                print(f"Found duplicate: {image_path}")
                if delete_duplicates:
                    os.remove(image_path)
                    print(f"Deleted duplicate: {image_path}")
                continue

            if rename_files:
                ext = os.path.splitext(image_path)[1].lower().lstrip('.')
                new_filename = self.get_next_filename(dest_folder, label, ext)
            else:
                new_filename = os.path.basename(image_path)

            dest_path = os.path.join(dest_folder, new_filename)

            if move_files:
                os.rename(image_path, dest_path)
            else:
                shutil.copy2(image_path, dest_path)

            print('\n', f'{"Moved" if move_files else "Copied"} {image_path} to {dest_path}')

        if delete_empty_folders:
            self.delete_empty_folders(input_folder)

    def delete_empty_folders(self, input_folder):
        for root, dirs, files in os.walk(input_folder, topdown=False):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                folder_files = os.listdir(dir_path)
                if len(folder_files) == 1 and folder_files[0] == '.DS_Store':
                    ds_store_path = os.path.join(dir_path, '.DS_Store')
                    os.remove(ds_store_path)
                    print(f"Deleted .DS_Store file: {ds_store_path}")

                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
                    print(f"Deleted empty folder: {dir_path}")
