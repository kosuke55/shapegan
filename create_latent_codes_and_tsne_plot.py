import torch
import numpy as np
from model.autoencoder import Autoencoder, LATENT_CODE_SIZE
from util import device, ensure_directory
from tqdm import tqdm
from datasets import CSVVoxelDataset
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import Bbox

autoencoder = Autoencoder(is_variational=False)
autoencoder.load()
autoencoder.eval()

dataset = CSVVoxelDataset('data/color-name-volume-mapping-bc-primates.csv', 'data/sdf-volumes/**/*.npy')

print("The dataset contains {:d} samples.".format(len(dataset)))

data_loader = DataLoader(dataset, shuffle=False, batch_size=16, num_workers=8)

USE_VOLUME_NEURON = False

LATENT_CODES_FILENAME = 'data/latent_codes.npy'
LATENT_CODES_TSNE_FILENAME = 'data/latent_codes_tsne.npy'

ensure_directory('plot')
TSNE_FIGURE_FILENAME = 'plot/tsne.pdf'

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


### t-SNE

print("Calculating embedding...")
tsne = TSNE(n_components=2)
latent_codes_embedded = tsne.fit_transform(latent_codes)

np.save(LATENT_CODES_TSNE_FILENAME, latent_codes_embedded)
print('Saved tsne-embedded latent codes to {:s}.'.format(LATENT_CODES_TSNE_FILENAME))


### Plot

colors = np.zeros((len(dataset), 3))
for i in range(len(dataset)):
    color = dataset.get_row(i)[1]
    colors[i, 0] = int(color[3:5], 16) / 255
    colors[i, 1] = int(color[5:7], 16) / 255
    colors[i, 2] = int(color[7:], 16) / 255

sizes = np.array([float(dataset.get_row(i)[4]) for i in range(len(dataset))])
sizes = np.power(sizes, 1/3)
sizes = sizes / np.max(sizes) * 60

size_inches = 6
fig, ax = plt.subplots(1, figsize=(size_inches, size_inches))
plt.axis('off')
ax.set_position([0, 0, 1, 1])
ax.scatter(latent_codes_embedded[:, 0], latent_codes_embedded[:, 1], facecolors=colors, linewidths=0.5, edgecolors=(0.1, 0.1, 0.1, 1.0), s=sizes)

for i in range(len(dataset)):
    ax.text(latent_codes_embedded[i, 0], latent_codes_embedded[i, 1] -2.5, dataset.get_row(i)[2], size=4, horizontalalignment='center', verticalalignment='top')

plt.savefig(TSNE_FIGURE_FILENAME, bbox_inches=Bbox([[0, 0], [size_inches, size_inches]]))
print("Saved t-SNE figure to {:s}.".format(TSNE_FIGURE_FILENAME))