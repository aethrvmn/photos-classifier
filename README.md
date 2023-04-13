
# Photo Classifier

A Python-based command-line tool for organizing and managing a collection of images. This tool classifies images based on their content, checks for duplicates, and organizes them into folders according to their categories. It supports various image formats, such as PNG, JPEG, TIFF, BMP, and GIF.

## Features

-   Classify images using a pre-trained DenseNet-121 model
-   Organize images into folders based on their categories
-   Check for and handle duplicate images
-   Rename image files based on their categories
-   Convert non-JPEG images to JPEG format
-   Remove empty folders

## Requirements

-   Python 3.7+
-   `torch` and `torchvision` packages
-   `Pillow` package
-   `numpy` package
-   `scikit-image` package
-   `pyheif` package (optional, for HEIC image support)
-   `tqdm` package (optional, for progress bars)

You can install the required packages using `pip`:

```bash
pip install torch torchvision Pillow numpy scikit-image pyheif tqdm
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
classifier.process_images(input_folder, output_folder, rename_files=True, check_duplicates=True, move_files=True, delete_duplicates=True, delete_empty_folders=True)
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

With this custom mapping, the script will use the new category names instead of the original ones provided by the DenseNet-121 model.

## Troubleshooting

If you encounter issues related to SSL certificate verification, you can disable SSL verification by passing `ssl_unverified=True` to the `ImageClassifier` constructor:

```python
classifier = ImageClassifier(ssl_unverified=True)
```
