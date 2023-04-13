
# Photo Classifier

A Python script that classifies and sorts photos into folders based on their content using the ResNeXt-101 32x48d WSL model. The model is pretrained on ImageNet, and you can customize the classification by providing a custom mapping of ImageNet classes to your desired categories.

## Requirements

-   PyTorch
-   torchvision
-   PIL (Pillow)
-   tqdm

## Installation
1.  Install the required libraries using pip:

```bash
pip install torch torchvision pillow tqdm
```

2.  Download the `photo_classifier.py` script.

## Usage

1.  Edit the `photo_classifier.py` script to set the `input_folder` and `output_folder` variables to the paths of your input images and the desired output folder, respectively.
2.  Update the `custom_mapping` dictionary in the script to map the ImageNet labels to your custom categories.
3.  Run the script:

```bash
python photo_classifier.py
```

4. Organize your input images into a folder. This folder can have subfolders with images as well. The script will search for images recursively.

5.  Run the script:

```bash
python photo_classifier.py
```

The script will classify the images and move them to the specified output folder, organizing them into subfolders based on the custom categories defined in `custom_mapping`.

## Customization

You can customize the image classifier by using different pre-trained models available in PyTorch. To use a different model, replace the model loading line in the `photo_classifier.py` script:

```python
model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl')
```

### Custom Labels

To create your own custom labels using a dictionary, update the `custom_mapping` dictionary in the `photo_classifier.py` script. The keys in the dictionary represent your custom labels, while the values are lists of ImageNet labels that correspond to the custom label.

For example, if you want to create custom labels for "sea", "forest", "animal", "people", "portrait", and "city", you can define the `custom_mapping` as follows:

```python
custom_mapping = {
    "sea": ["beach", "seashore", "coast", "ocean", "sea"],
    "forest": ["forest", "wood", "woods", "jungle"],
    "animal": ["dog", "cat", "bird", "fish", "lion", "tiger", "elephant", "zebra"],
    "people": ["person", "man", "woman", "boy", "girl"],
    "portrait": ["face", "head", "selfie"],
    "city": ["cityscape", "skyscraper", "buildings", "street"]
}
```
Please note that this example contains only a small subset of possible ImageNet labels. You'll need to expand this list based on the actual labels from ImageNet that you want to map to your custom categories.

## Notes

-   The script ignores hidden files and non-image files.
-   The input images will be moved, not copied, to the output folder.
-   If you want to use a different model, replace the `model` variable with the desired model and adjust the preprocessing steps accordingly.

## Limitations

The accuracy of the classification and organization depends on the pre-trained model used and the similarity of your images to the ImageNet dataset. You might need to fine-tune the model or use a different pre-trained model for better accuracy in some cases.





## Usage

1. Organize your input images into a folder. This folder can have subfolders with images as well. The script will search for images recursively.

2. Update the `custom_mapping` dictionary in the script to map ImageNet classes to your desired categories. For example:

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
```

3.  Set the `input_folder` and `output_folder` variables in the script:

python

`input_folder = 'photos/input'
output_folder = 'photos/output'`

4.  Run the script:

`python photo_classifier.py`

The script will classify the images and move them to the specified output folder, organizing them into subfolders based on the custom categories defined in `custom_mapping`.

