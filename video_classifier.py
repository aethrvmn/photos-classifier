import ffmpeg
import json
import moviepy.editor
import os
import shutil
import ssl
import urllib
import urllib.request
import random
import imagehash
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional
import moviepy
import imageio
import subprocess

from PIL import Image
from moviepy.editor import VideoFileClip, concatenate_videoclips
from tqdm import tqdm
from torchvision.io import read_video
from torchvision.transforms import Compose, Lambda
from torchvision.transforms.functional import normalize
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample



class VideoClassifier:

    def __init__(self, ssl_unverified=False, custom_mapping=None):
        if ssl_unverified and hasattr(ssl, '_create_unverified_context'):
            ssl._create_default_https_context = ssl._create_unverified_context

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        self.model.eval()
        self.model.to(self.device)

        json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
        json_filename = "kinetics_classnames.json"
        urllib.request.urlretrieve(json_url, json_filename)

        with open(json_filename, "r") as f:
            kinetics_classnames = json.load(f)

        self.kinetics_id_to_classname = {v: str(k).replace('"', "") for k, v in kinetics_classnames.items()}


        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        num_frames = 8
        sampling_rate = 8
        frames_per_second = 30

        self.transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(size=side_size),
                    CenterCropVideo(crop_size=(crop_size, crop_size))
                ]
            ),
        )


        self.clip_duration = (num_frames * sampling_rate) / frames_per_second
        self.custom_mapping = custom_mapping if custom_mapping else {}

    def uniform_temporal_subsample(self, x: torch.Tensor, num_samples: int, temporal_dim: int = -3):

        t = x.shape[temporal_dim]
        assert num_samples > 0 and t > 0
        # Sample by nearest neighbor interpolation if num_samples > t.
        indices = torch.linspace(0, t - 1, num_samples)
        indices = torch.clamp(indices, 0, t - 1).long()
        return torch.index_select(x, temporal_dim, indices)

    def classify_video(self, video_path, input_video=None):

        frame_rate = input_video.fps
        start_frame = 0
        end_frame = int(self.clip_duration * frame_rate)
        video_data = (input_video.get_frame(i / frame_rate) for i in range(start_frame, end_frame))
            
        video_data = np.stack(list(video_data), axis=0)  # Stack frames into a single array
        video_data = torch.from_numpy(video_data).permute(3, 0, 1, 2)  # Change tensor shape to (C, T, H, W)

        video_data = self.transform({"video": video_data})

        inputs = video_data["video"]
        inputs = inputs.to(self.device)

        with torch.no_grad():
            output = self.model(inputs[None, ...])

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top1_prob, top1_catid = torch.topk(probabilities, 1)
        top1_catid_int = top1_catid[0].item()
        category = self.custom_mapping.get(self.kinetics_id_to_classname[top1_catid_int], self.kinetics_id_to_classname[top1_catid_int])
        input_video.close()

        return category, top1_prob.item()


    def video_similarity(self, video1_path, video2_path, frame_interval=1):
        print(f"Comparing {video1_path} to {video2_path}")
        video1 = VideoFileClip(video1_path)
        video2 = VideoFileClip(video2_path)

        video1_frames = [Image.fromarray(frame) for i, frame in enumerate(video1.iter_frames()) if i % frame_interval == 0]
        video2_frames = [Image.fromarray(frame) for i, frame in enumerate(video2.iter_frames()) if i % frame_interval == 0]

        video1_hashes = [imagehash.phash(frame) for frame in video1_frames]
        video2_hashes = [imagehash.phash(frame) for frame in video2_frames]

        similarities = []
        for hash1 in video1_hashes:
            frame_similarities = [hash1 - hash2 for hash2 in video2_hashes]
            similarities.append(min(frame_similarities))

        avg_similarity = sum(similarities) / len(similarities)

        video1.close()
        video2.close()

        return avg_similarity


    def convert_to_mp4(self, video_path):
        input_video = VideoFileClip(video_path)
        video_format = ffmpeg.probe(video_path)["streams"][0]["codec_name"]

        if video_format.lower() == 'h264':
            return video_path, input_video
        else:
            filename = os.path.basename(video_path)
            new_video_path = os.path.join('photos', 'converted', os.path.splitext(filename)[0] + '.mp4')
            os.makedirs(os.path.dirname(new_video_path), exist_ok=True)

            input_video.write_videofile(new_video_path, codec='libx264')
            input_video.close()
            new_input_video = VideoFileClip(new_video_path, 'ffmpeg')
            return new_video_path, new_input_video


    def is_duplicate_video(self, video_path, output_folder, label, threshold=10, length_tolerance=0.1):
        dest_folder = os.path.join(output_folder, label)
        if not os.path.exists(dest_folder):
            return False

        input_video_reader = VideoFileClip(video_path)
        input_video_duration = input_video_reader.duration
        input_video_frame_size = input_video_reader.size

        for existing_file in os.listdir(dest_folder):
            if existing_file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv')) and not existing_file.startswith('.'):
                existing_video_path = os.path.join(dest_folder, existing_file)
                existing_video_reader = VideoFileClip(existing_video_path)
                existing_video_duration = existing_video_reader.duration
                existing_video_frame_size = existing_video_reader.size

                if (abs(input_video_duration - existing_video_duration) <= length_tolerance * input_video_duration) and (input_video_frame_size == existing_video_frame_size):
                    similarity = self.video_similarity(video_path, existing_video_path)
                    if similarity <= threshold:
                        return True

        return False


    def open_video_with_warning(self, video_path):
        try:
            return VideoFileClip(video_path, 'ffmpeg')
        except Exception as e:
            print(f"Warning: Unable to open video: {video_path}. Error: {str(e)}")
            return None

    def get_next_filename(self, folder, prefix, ext):
        i = 1
        while True:
            filename_underscore = f"{prefix}_{i}.{ext}"
            filename_no_underscore = f"{prefix}{i}.{ext}"
            if not os.path.exists(os.path.join(folder, filename_underscore)) and not os.path.exists(os.path.join(folder, filename_no_underscore)):
                return filename_no_underscore
            i += 1

    def get_video_files_recursively(self, input_folder):

        video_files = [ os.path.join(root, file) for root, _, files in os.walk(input_folder) for file in files if file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv')) and not file.startswith('.')]
        random.shuffle(video_files)
        return video_files


    def process_videos(self, input_folder, output_folder, check_duplicates=True, move_files=True, delete_duplicates=True, delete_empty_folders=True, convert_to_mp4=True, rename_files=True, min_duration = 5):
        video_files = self.get_video_files_recursively(input_folder)

        for video_path in tqdm(video_files):
            try:
                print(f"Started work on {video_path}...")
                is_mp4 = video_path.lower().endswith('.mp4')

                if convert_to_mp4:
                    video_path, input_video = self.convert_to_mp4(video_path)
                else:
                    input_video = self.open_video_with_warning(video_path)

                if input_video is None:
                    continue
                   
                if input_video.duration < min_duration:
                    print(f"Video {video_path} is too short ({input_video.duration:.2f}s), skipping")
                    continue

                label, _ = self.classify_video(video_path, input_video=input_video)
                print(f"Classification of {video_path} complete")
                input_video.close()
                del input_video
                dest_folder = os.path.join(output_folder, label)
                os.makedirs(dest_folder, exist_ok=True)
                print("Checking for duplicates...")
                if check_duplicates and self.is_duplicate_video(video_path, output_folder, label):
                    print(f"Found duplicate: {video_path}")
                    if delete_duplicates:
                        os.remove(video_path)
                        print(f"Deleted duplicate: {video_path}")
                    continue
                else:
                    print("No duplicates found")

                if rename_files:
                    ext = os.path.splitext(video_path)[1].lower().lstrip('.')
                    new_filename = self.get_next_filename(dest_folder, label, ext)
                else:
                    new_filename = os.path.basename(video_path)

                dest_path = os.path.join(dest_folder, new_filename)

                if move_files:
                    os.rename(video_path, dest_path)
                else:
                    shutil.copy2(video_path, dest_path)

                print('\n', f'{"Moved" if move_files else "Copied"} {video_path} to {dest_path}')
            except:
                continue
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



