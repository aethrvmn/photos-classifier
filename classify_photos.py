# image_classifier_usage.py
from classifier import ImageClassifier

input_folder = 'photos/input'
output_folder = 'photos/output'

# Define your custom labels here
custom_labels = {
    # Add your custom mappings here
}

classifier = ImageClassifier(ssl_unverified=True, custom_mapping=custom_labels)
classifier.process_images(input_folder, output_folder, rename_files=True, check_duplicates=True, move_files=True, delete_duplicates=True, delete_empty_folders=True)
