# classify_videos.py
from video_classifier import VideoClassifier

input_folder = 'photos/input'
output_folder = 'photos/videos'

# Define your custom labels here
custom_labels = {
    # Add your custom mappings here
}

classifier = VideoClassifier(ssl_unverified=True, custom_mapping=custom_labels)
classifier.process_videos(input_folder, output_folder, rename_files=True, check_duplicates=True, move_files=True, delete_duplicates=True, delete_empty_folders=True, convert_to_mp4=True)
