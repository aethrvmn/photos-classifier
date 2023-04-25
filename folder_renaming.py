import os
import shutil
from image_classifier import ImageClassifier
from tqdm import tqdm

def rename_images_based_on_folder(classifier, input_folder, output_folder):
    image_files = classifier.get_image_files_recursively(input_folder)

    for image_path in tqdm(image_files):
        folder_name = os.path.basename(os.path.dirname(image_path))
        dest_folder = os.path.join(output_folder, folder_name)
        os.makedirs(dest_folder, exist_ok=True)

        ext = os.path.splitext(image_path)[1].lower().lstrip('.')
        new_filename = classifier.get_next_filename(dest_folder, folder_name, ext)
        dest_path = os.path.join(dest_folder, new_filename)

        shutil.copy2(image_path, dest_path)
        print('\n', f'Copied {image_path} to {dest_path}')

input_folder = "input"  # Change this to the path of your input folder
output_folder = "output"  # Change this to the path of your output folder

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize the ImageClassifier
classifier = ImageClassifier()

# Rename the images based on the folder name without using the classifier
rename_images_based_on_folder(classifier, input_folder, output_folder)
