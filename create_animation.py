from util import device, ensure_directory
from datasets import CSVVoxelDataset
from torch.utils.data import DataLoader
import scipy
import numpy as np
from rendering import MeshRenderer
import torch
from tqdm import tqdm
import cv2
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.offsetbox import Bbox
from sklearn.cluster import KMeans
import os
from PIL import ImageFont


SAMPLE_COUNT = 16 # Number of distinct objects to generate and interpolate between
TRANSITION_FRAMES = 60

FRAMES = SAMPLE_COUNT * TRANSITION_FRAMES
progress = np.arange(FRAMES, dtype=float) / TRANSITION_FRAMES

dataset = CSVVoxelDataset('data/color-name-volume-binomial-mapping-bc-primates-and-humans.csv', 'data/sdf-volumes/**/*.npy')

USE_VOLUME_NEURON = True

font = ImageFont.truetype('helvetica.ttf', 60)
#font = ImageFont.truetype('DejaVu Sans.ttf', 60)

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

print("Calculating embedding...")
tsne = TSNE(n_components=2)
latent_codes_embedded = tsne.fit_transform(latent_codes)
print("Calculating clusters...")
kmeans = KMeans(n_clusters=SAMPLE_COUNT)

indices = np.zeros(SAMPLE_COUNT, dtype=int)
kmeans_clusters = kmeans.fit_predict(latent_codes_embedded)
for i in range(SAMPLE_COUNT):
    center = kmeans.cluster_centers_[i, :]
    dist = np.linalg.norm(latent_codes_embedded - center[np.newaxis, :], axis=1)
    indices[i] = np.argmin(dist)

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math

names = [dataset.get_row(i)[2] for i in range(len(dataset))]

def try_find_shortest_roundtrip(indices):
    best_order = indices
    best_distance = None
    for _ in range(5000):
        candiate = best_order.copy()
        a = random.randint(0, SAMPLE_COUNT-1)
        b = random.randint(0, SAMPLE_COUNT-1)
        candiate[a] = best_order[b]
        candiate[b] = best_order[a]
        dist = np.sum(np.linalg.norm(latent_codes_embedded[candiate, :] - latent_codes_embedded[np.roll(candiate, 1), :], axis=1)).item()
        if best_distance is None or dist < best_distance:
            best_distance = dist
            best_order = candiate

    return best_order, best_distance

def find_shortest_roundtrip(indices):
    best_order, best_distance = try_find_shortest_roundtrip(indices)

    for _ in tqdm(range(100)):
        np.random.shuffle(indices)
        order, distance = try_find_shortest_roundtrip(indices)
        if distance < best_distance:
            best_order = order
    return best_order

print("Calculating trip...")
indices = find_shortest_roundtrip(indices)
indices = np.concatenate((indices, indices[0][np.newaxis]))

SIZE = latent_codes.shape[0]

stop_latent_codes = latent_codes[indices, :]
stop_names = [names[i] for i in indices]
stop_names.append(stop_names[0])

spline = scipy.interpolate.CubicSpline(np.arange(SAMPLE_COUNT + 1), stop_latent_codes, axis=0, bc_type='periodic')
frame_latent_codes = spline(progress)

embedded_spline = scipy.interpolate.CubicSpline(np.arange(SAMPLE_COUNT + 1), latent_codes_embedded[indices, :], axis=0, bc_type='periodic')
frame_latent_codes_embedded = embedded_spline(progress)
frame_latent_codes_embedded[0, :] = frame_latent_codes_embedded[-1, :]

colors = dataset.get_colors()

color_spline = scipy.interpolate.CubicSpline(np.arange(SAMPLE_COUNT + 1), colors[indices, :], axis=0, bc_type='periodic')
frame_colors = color_spline(progress)
frame_colors = np.clip(frame_colors, 0, 1)

frame_colors = np.zeros((progress.shape[0], 3))
for i in range(SAMPLE_COUNT):
    frame_colors[i*TRANSITION_FRAMES:(i+1)*TRANSITION_FRAMES, :] = np.linspace(colors[indices[i]], colors[indices[i+1]], num=TRANSITION_FRAMES)

width, height = 40, 40

PLOT_FILE_NAME = 'tsne.png'
ensure_directory('images')

margin = 4
range_x = (latent_codes_embedded[:, 0].min() - margin, latent_codes_embedded[:, 0].max() + margin)
range_y = (latent_codes_embedded[:, 1].min() - margin, latent_codes_embedded[:, 1].max() + margin)

plt.ioff()

def create_plot(index, resolution=1080, filename=PLOT_FILE_NAME, dpi=100):
    frame_color = frame_colors[index, :]
    frame_color = (frame_color[0], frame_color[1], frame_color[2], 1.0)

    size_inches = resolution / dpi

    fig, ax = plt.subplots(1, figsize=(size_inches, size_inches), dpi=dpi)
    ax.set_position([0, 0, 1, 1])
    plt.axis('off')
    ax.set_xlim(range_x)
    ax.set_ylim(range_y)

    ax.plot(frame_latent_codes_embedded[:, 0], frame_latent_codes_embedded[:, 1], c=(0.2, 0.2, 0.2, 1.0), zorder=1, linewidth=2) # black line
    ax.scatter(latent_codes_embedded[:, 0], latent_codes_embedded[:, 1], facecolors=colors, s = 40, zorder=0, linewidths=1, edgecolors=(0.1, 0.1, 0.1, 1.0)) # all dots
    ax.scatter(frame_latent_codes_embedded[index, 0], frame_latent_codes_embedded[index, 1], facecolors=frame_color, s = 400, linewidths=2, edgecolors=(0.1, 0.1, 0.1, 1.0), zorder=2) # current position
    ax.scatter(latent_codes_embedded[indices, 0], latent_codes_embedded[indices, 1], facecolors=colors[indices, :], s = 280, linewidths=1, edgecolors=(0.1, 0.1, 0.1, 1.0), zorder=3) # stops
    
    fig.savefig(filename, bbox_inches=Bbox([[0, 0], [size_inches, size_inches]]), dpi=dpi)
    plt.close(fig)

frame_latent_codes = torch.tensor(frame_latent_codes, dtype=torch.float32, device=device)

print("Rendering...")
viewer = MeshRenderer(size=1080, start_thread=False)
viewer.rotation = (130+180, 20)

def render_frame(frame_index):
    viewer.model_color = frame_colors[frame_index, :]
    with torch.no_grad():
        viewer.set_voxels(autoencoder.decode(frame_latent_codes[frame_index, :]))
    image_mesh = viewer.get_image(flip_red_blue=True)

    progress, model_index = math.modf(frame_index / TRANSITION_FRAMES + 0.5)
    img = Image.fromarray(np.uint8(image_mesh))
    name = stop_names[int(model_index)]
    d = ImageDraw.Draw(img)
    width, _ = d.textsize(name, font=font)
    color = int(max(0, abs(progress - 0.5) * 2 * 3 - 2) * 255)
    d.text((540 - width // 2, 980), name, font=font, fill=(color, color, color))
    image_mesh = np.array(img)[:, :, :3]

    create_plot(frame_index)
    image_tsne = plt.imread(PLOT_FILE_NAME)[:, :, [2, 1, 0]] * 255

    image = np.concatenate((image_mesh, image_tsne), axis=1)

    cv2.imwrite("images/frame-{:05d}.png".format(frame_index), image)


for frame_index in tqdm(range(SAMPLE_COUNT * TRANSITION_FRAMES)):
    render_frame(frame_index)
    frame_index += 1

print("\n\nUse this command to create a video:\n")
print('ffmpeg -framerate 30 -i images/frame-%05d.png -c:v libx264 -profile:v high -crf 19 -pix_fmt yuv420p video.mp4')