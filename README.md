
# Image Organizer

Image Organizer is a Python script that utilizes a pre-trained ResNet-18 model from PyTorch to classify and organize images into custom categories. This script is useful for organizing large collections of images into more manageable and meaningful groups.

## Requirements

-   PyTorch
-   torchvision
-   PIL (Pillow)

## Installation
1.  Install the required libraries using pip:

```bash
pip install torch torchvision pillow
```

2.  Download the `photo_classifier.py` script.

## Usage

1.  Edit the `photo_classifier.py` script to set the `input_folder` and `output_folder` variables to the paths of your input images and the desired output folder, respectively.
2.  Update the `custom_mapping` dictionary in the script to map the ImageNet labels to your custom categories.
3.  Run the script:

```bash
python photo_classifier.py
```

4.  The script will process each image in the `input_folder`, classify it using the pre-trained ResNet-18 model, and move it to a subfolder in the `output_folder` based on the custom category mapping.

## Customization

You can customize the image classifier by using different pre-trained models available in PyTorch, such as ResNet-34, ResNet-50, ResNet-101, or ResNet-152. To use a different model, replace the model loading line in the `photo_classifier.py` script:

```python
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
```

Replace '*resnet34*' with the desired model name.

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

## Limitations

The accuracy of the classification and organization depends on the pre-trained model used and the similarity of your images to the ImageNet dataset. You might need to fine-tune the model or use a different pre-trained model for better accuracy in some cases.
