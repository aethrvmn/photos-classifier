# Photo Classifier

A Python-based command-line tool for organizing and managing a collection of images. This tool classifies images based on their content, checks for duplicates, and organizes them into folders according to their categories. It supports various image formats, such as PNG, JPEG, TIFF, BMP, and GIF.

## Features

-   Classify images using a pre-trained Vision Transformer model
-   Organize images into folders based on their categories
-   Check for and handle duplicate images
-   Rename image files based on their categories
-   Convert non-JPEG images to JPEG format
-   Remove empty folders

## Requirements

-   Python 3.7+
-   torch and torchvision packages
-   Pillow package
-   numpy package
-   scikit-image package
-   timm package
-   tqdm package (optional, for progress bars)

You can install the required packages using pip:

```bash
pip install torch torchvision Pillow numpy scikit-image timm tqdm
```

## Usage

First, create a Python script (e.g., `classify_photos.py`) that imports the `ImageClassifier` class from the provided code file (e.g., `classifier.py`), and then create an instance of the class. After that, call the `process_images` method with the desired input and output folder paths.

```python
from classifier import ImageClassifier

input_folder = 'photos/input'
output_folder = 'photos/output'

# Define your custom labels here
custom_labels = {
    # Add your custom mappings here
}

classifier = ImageClassifier(custom_mapping=custom_labels)
classifier.process_images(input_folder, output_folder, rename_files=True, check_duplicates=True, move_files=True, delete_duplicates=True, delete_empty_folders=True, convert_to_jpeg=True)

```

You can customize the behavior of the `process_images` method using the following optional parameters:

-   `rename_files`: Set to `True` (default) to rename image files based on their categories, or `False` to keep the original filenames.
-   `check_duplicates`: Set to `True` (default) to check for and handle duplicate images, or `False` to skip duplicate checking.
-   `move_files`: Set to `True` (default) to move image files from the input folder to the output folder, or `False` to copy them instead.
-   `delete_duplicates`: Set to `True` (default) to delete duplicate images, or `False` to keep them.
-   `delete_empty_folders`: Set to `True` (default) to delete empty folders in the input directory, or `False` to keep them.
-   `convert_to_jpeg`: Set to `True` (default) to convert non-JPEG images to JPEG format, or `False` to keep the original formats.

## Customizations

You can create a custom mapping of categories by providing a dictionary to the `ImageClassifier` constructor:

```python
custom_mapping = {
    "sea": ["beach", "seashore", "coast", "ocean", "sea"],
    "forest": ["forest", "wood", "woods", "jungle"],
    "animal": ["dog", "cat", "bird", "fish", "lion", "tiger", "elephant", "zebra"],
    "people": ["person", "man", "woman", "boy", "girl"],
    "portrait": ["face", "head", "selfie"],
    "city": ["cityscape", "skyscraper", "buildings", "street"],
    "nature": ["cliff", "alp", "barn", "box_turtle"]
}

classifier = ImageClassifier(custom_mapping=custom_mapping)
```

With this custom mapping, the script will use the new category names instead of the original ones providedby the Vision Transformer model.

## Troubleshooting

If you encounter issues related to SSL certificate verification, you can disable SSL verification by passing `ssl_unverified=True` to the `ImageClassifier` constructor:

```python
classifier = ImageClassifier(ssl_unverified=True)
```

## Code Overview

The provided `classifier.py` script contains the `ImageClassifier` class that handles image classification, duplicate checking, and file management. The class uses a pre-trained Vision Transformer model from the `timm` package to classify images. It also utilizes the `Pillow` package for image processing and the `scikit-image` package for calculating image similarity.

### Key Methods

-   `__init__(self, ssl_unverified=False, custom_mapping=None)`: Initializes the `ImageClassifier` instance with optional SSL verification and custom category mapping.
-   `classify_image(self, image_path, input_image=None)`: Classifies an image given its file path or an opened image instance.
-   `calculate_image_similarity(self, img1, img2)`: Calculates the similarity between two images using the Structural Similarity Index (SSIM).
-   `is_duplicate(self, image, output_folder, label, threshold=0.95)`: Checks if an image is a duplicate based on its similarity to existing images in the output folder.
-   `process_images(self, input_folder, output_folder, rename_files=True, check_duplicates=True, move_files=True, delete_duplicates=True, delete_empty_folders=True, convert_to_jpeg=True)`: Processes images in the input folder, classifies them, and organizes them in the output folder.

### Additional Methods

The class also contains several helper methods for image processing and file management, such as:

-   `get_image_files_recursively(self, input_folder)`: Retrieves image files recursively from the input folder.
-   `convert_to_rgb(self, image)`: Converts an image to the RGB color space.
-   `get_next_filename(self, folder, prefix, ext)`: Generates the next available filename for an image based on its category.
-   `convert_to_jpeg(self, image_path)`: Converts a non-JPEG image to JPEG format and deletes the original file.
-   `open_image_with_warning(self, image_path)`: Opens an image file and warns if the image is unidentified or corrupted.
-   `delete_empty_folders(self, input_folder)`: Recursively deletes empty folders in the input folder.
