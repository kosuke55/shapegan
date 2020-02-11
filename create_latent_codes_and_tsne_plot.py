import torch
import numpy as np
from model.autoencoder import Autoencoder, LATENT_CODE_SIZE
from util import device, ensure_directory
from tqdm import tqdm
from datasets import *
from torch.utils.data import DataLoader

autoencoder = Autoencoder(is_variational=False)
autoencoder.load()
autoencoder.eval()

dataset = CSVVoxelDataset('data/color-name-volume-mapping-bc-primates.csv', 'data/sdf-volumes/**/*.npy')

print("The dataset contains {:d} samples.".format(len(dataset)))

data_loader = DataLoader(dataset, shuffle=False, batch_size=16, num_workers=8)

USE_VOLUME_NEURON = False

LATENT_CODES_FILENAME = 'data/latent_codes.npy'

latent_codes = np.zeros((len(dataset), LATENT_CODE_SIZE))

with torch.no_grad():
    p = 0
    for batch in tqdm(data_loader, desc='Creating latent codes'):
        voxels, meta = batch

        if USE_VOLUME_NEURON:
            volume_data = [float(v) for v in meta[4]]
        
        batch_latent_codes = autoencoder.encode(voxels.to(device), volume=volume_data if USE_VOLUME_NEURON else None).cpu().numpy()

        latent_codes[p:p+batch_latent_codes.shape[0]] = batch_latent_codes
        p += batch_latent_codes.shape[0]

np.save(LATENT_CODES_FILENAME, latent_codes)
print('Saved latent codes to {:s}.'.format(LATENT_CODES_FILENAME))