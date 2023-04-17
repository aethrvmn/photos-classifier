import os
import ssl

class BaseClassifier:
    def __init__(self, ssl_unverified=False):
        if ssl_unverified and hasattr(ssl, '_create_unverified_context'):
            ssl._create_default_https_context = ssl._create_unverified_context

    def get_files_recursively(self, input_folder, file_extensions):
        return [os.path.join(root, file) for root, _, files in os.walk(input_folder) for file in files if
            file.lower().endswith(file_extensions) and not file.startswith('.')]

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

    def get_next_filename(self, folder, prefix, ext):
        i = 1
        while True:
            filename_underscore = f"{prefix}_{i}.{ext}"
            filename_no_underscore = f"{prefix}{i}.{ext}"
            if not os.path.exists(os.path.join(folder, filename_underscore)) and not os.path.exists(os.path.join(folder, filename_no_underscore)):
                return filename_no_underscore
            i += 1

    # Add other common methods here
