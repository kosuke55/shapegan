from model.sdf_net import SDFNet, LATENT_CODE_SIZE, LATENT_CODES_FILENAME
from util import device, standard_normal_distribution, ensure_directory
from dataset import dataset
import scipy
import numpy as np
from rendering import MeshRenderer
import time
import torch
from tqdm import tqdm
import cv2
import random
import sys
import os

SAMPLE_COUNT = 30 # Number of distinct objects to generate and interpolate between
TRANSITION_FRAMES = 60

from dataset import dataset as dataset
dataset.load_voxels(device)

by_size = open('data/by_size.txt').readlines()
by_size = [i.strip() for i in by_size]

DIRECTORY_MODELS = 'data/meshes/'
MODEL_EXTENSION = '.ply'

def get_model_files():
    for directory, _, files in os.walk(DIRECTORY_MODELS):
        for filename in files:
            if filename.endswith(MODEL_EXTENSION):
                yield os.path.join(directory, filename)
filenames = sorted(list(get_model_files()))


indices = []
for name in by_size:
    for i, file_name in enumerate(filenames):
        if name in file_name:
            indices.append(i)
            break

voxels = dataset.voxels
from model.autoencoder import Autoencoder
autoencoder = Autoencoder(is_variational=False)
autoencoder.load()
autoencoder.eval()
with torch.no_grad():
    codes = autoencoder.encode(voxels).cpu().numpy()

codes = codes[indices, :]
SAMPLE_COUNT = codes.shape[0]

spline = scipy.interpolate.CubicSpline(np.arange(SAMPLE_COUNT), codes, axis=0)

def create_image_sequence():
    ensure_directory('images')
    frame_index = 0
    viewer = MeshRenderer(size=1080, start_thread=False)
    progress_bar = tqdm(total=SAMPLE_COUNT * TRANSITION_FRAMES)

    for sample_index in range(SAMPLE_COUNT):
        for step in range(TRANSITION_FRAMES):
            code = torch.tensor(spline(float(sample_index) + step / TRANSITION_FRAMES), dtype=torch.float32, device=device)
            with torch.no_grad():
                viewer.set_voxels(autoencoder.decode(code))
            image = viewer.get_image(flip_red_blue=True)
            cv2.imwrite("images/frame-{:05d}.png".format(frame_index), image)
            frame_index += 1
            progress_bar.update()
    
    print("\n\nUse this command to create a video:\n")
    print('ffmpeg -framerate 30 -i images/frame-%05d.png -c:v libx264 -profile:v high -crf 19 -pix_fmt yuv420p video.mp4')

create_image_sequence()