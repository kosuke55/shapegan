import torch
import numpy as np
from model.autoencoder import Autoencoder
from dataset import dataset as dataset
from util import device, ensure_directory
from tqdm import tqdm
import skimage.measure
import trimesh

ensure_directory('data/reconstructed/')

dataset.load_voxels(device)

autoencoder = Autoencoder(is_variational=False)
autoencoder.load()
autoencoder.eval()


with torch.no_grad():
    for i in tqdm(range(dataset.voxels.shape[0])):
        voxels = autoencoder(dataset.voxels[i, :, :, :])[0].detach().cpu().numpy()
        voxel_resolution = voxels.shape[1]
        voxels = np.pad(voxels, 1, mode='constant', constant_values=1)
        size = 2
        spacing = size / voxel_resolution
        try:
            vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(voxels, level=0, spacing=(spacing, spacing, spacing))
        except ValueError:
            print("No sign changes in the volume. Not creating a mesh for item {:d}".format(i))
            continue
        
        vertices -= size / 2
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        mesh.export('data/reconstructed/{:04d}.ply'.format(i))