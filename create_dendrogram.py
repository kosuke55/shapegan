from util import device, ensure_directory
from datasets import CSVVoxelDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import os
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

dataset = CSVVoxelDataset('data/color-name-volume-mapping-bc-primates.csv', 'data/sdf-volumes/**/*.npy')

USE_VOLUME_NEURON = False

FIGURE_SIZE = 12

from model.autoencoder import Autoencoder, LATENT_CODE_SIZE
autoencoder = Autoencoder(is_variational=False)
autoencoder.load()
autoencoder.eval()

latent_codes = np.zeros((len(dataset), LATENT_CODE_SIZE))
data_loader = DataLoader(dataset, shuffle=False, batch_size=16, num_workers=8)

with torch.no_grad():
    p = 0
    for batch in tqdm(data_loader, desc='Creating latent codes'):
        voxels, meta = batch

        if USE_VOLUME_NEURON:
            volume_data = [float(v) for v in meta[4]]
        
        batch_latent_codes = autoencoder.encode(voxels.to(device), volume=volume_data if USE_VOLUME_NEURON else None).cpu().numpy()

        latent_codes[p:p+batch_latent_codes.shape[0]] = batch_latent_codes
        p += batch_latent_codes.shape[0]

names = [row[2] for row in dataset.rows]

colors = [row[1] for row in dataset.rows]


def link_color_func(id):
    if id < len(colors):
        return colors[id]
    else:
        return 'k'

X = pdist(latent_codes, metric='euclidean')
Z = linkage(X)
fig = plt.figure(figsize=(FIGURE_SIZE, FIGURE_SIZE))
ax = plt.gca()
ax.xaxis.labelpad = 20
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

print(Z.shape, len(colors))

for i in range(Z.shape[0]):
    a, b = int(Z[i, 0]), int(Z[i, 1])
    if colors[a] == colors[b]:
        colors.append(colors[a])
    else:
        colors.append('#999999')

dn = dendrogram(Z, labels=names, orientation='left', link_color_func=link_color_func, leaf_font_size=10)
fig.subplots_adjust(right=0.75)
plt.savefig("plots/dendrogram.pdf")