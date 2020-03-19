import dendropy
from util import device
from datasets import CSVVoxelDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

tree = dendropy.Tree.get_from_path("data/consensusTree_10kTrees_Primates_Version3.nex", "nexus")
pdm = dendropy.PhylogeneticDistanceMatrix()
pdm.compile_from_tree(tree)
taxon_namespace = tree.taxon_namespace

dataset = CSVVoxelDataset('data/color-name-volume-binomial-mapping-bc-primates-and-humans.csv', 'data/sdf-volumes/**/*.npy')

species = [row[10].replace('_', ' ') for row in dataset.rows]
names = [row[2] for row in dataset.rows]

indices = []
taxons = []

for i, name in enumerate(species):
    current_taxons = taxon_namespace.findall(name)
    if len(current_taxons) > 0:
        indices.append(i)
        taxons.append(current_taxons[0])

colors = [row[1] for row in dataset.rows]
colors = [colors[i] for i in indices]
names = [names[i] for i in indices]

USE_VOLUME_NEURON = False

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

latent_codes = latent_codes[indices, :]

latent_distances = []
phylogenetic_distances = []

latent_distances = []
phylogenetic_distances = []

for i in range(len(indices)):
    for j in range(i):
        latent_distances.append(np.linalg.norm(latent_codes[i, :] - latent_codes[j, :]))
        phylogenetic_distances.append(pdm.distance(taxons[i], taxons[j]))
        
plt.scatter(latent_distances, phylogenetic_distances, s=0.5, marker='o', )
plt.xlabel('Latent space distance')
plt.ylabel('Phylogenetic distance')
plt.savefig('plots/pairs-16-cosine.pdf')