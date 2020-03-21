import csv
import torch
import numpy as np
from model.autoencoder import Autoencoder, LATENT_CODE_SIZE
from util import device, ensure_directory
from tqdm import tqdm
from datasets import CSVVoxelDataset
import cv2
from rendering import MeshRenderer

autoencoder = Autoencoder(is_variational=False)
autoencoder.load()
autoencoder.eval()

viewer = MeshRenderer(size=1080, start_thread=False)
viewer.rotation = (130+180, 20)
viewer.model_color = (0.6, 0.6, 0.6)

with open('data/primate_brain_latent_code_aces_w_human_20_03_2020.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    csv_iterator = iter(csv_reader)
    next(csv_iterator)
    csv_rows = list(csv_iterator)

for row in tqdm(csv_rows):
    node = int(row[0])
    latent_code = torch.tensor([float(v) for v in row[1:]], dtype=torch.float32, device=device)

    with torch.no_grad():
        voxels = autoencoder.decode(latent_code)
    viewer.set_voxels(voxels)
    image = viewer.get_image(flip_red_blue=True)
    cv2.imwrite("reconstructed/{:d}.png".format(node), image)