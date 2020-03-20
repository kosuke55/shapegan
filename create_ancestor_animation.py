import dendropy
from util import device
from datasets import CSVVoxelDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
from rendering import MeshRenderer
import cv2
from PIL import Image, ImageDraw, ImageFont

USE_VOLUME_NEURON = True

FRAME_COUNT = 15 * 30
STILL_FRAMES = 2 * 30

tree = dendropy.Tree.get_from_path("data/consensusTree_10kTrees_Primates_Version3.nex", "nexus")
tree.calc_node_ages()

start_taxon = tree.taxon_namespace.findall("Cebus apella")[0]
print(start_taxon)

start_node = tree.find_node_for_taxon(start_taxon)

stop_nodes = [start_node]
current_node = start_node

while current_node.parent_node is not None:
    current_node = current_node.parent_node
    stop_nodes.append(current_node)

leafs = []
non_leafs = []

for node in tree.preorder_node_iter():
    if node.is_leaf():
        leafs.append(node)
    else:
        non_leafs.append(node)

nodes_by_id = leafs + non_leafs

def get_node_id(node):
    return nodes_by_id.index(node) + 1

stop_ids = [get_node_id(node) for node in stop_nodes]

stop_ages = [node.age for node in stop_nodes]
max_age = max(stop_ages)
relative_ages = [age / max_age for age in stop_ages]

latent_codes_by_node_id = {}

with open('data/primate_brain_latent_code_aces_w_human_20_03_2020.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    csv_iterator = iter(csv_reader)
    next(csv_iterator)
    csv_rows = list(csv_iterator)

    for row in csv_rows:
        node = int(row[0])
        latent_code = torch.tensor([float(v) for v in row[1:]], dtype=torch.float32, device=device)
        latent_codes_by_node_id[node] = latent_code

dataset = CSVVoxelDataset('data/color-name-volume-binomial-mapping-bc-primates-and-humans.csv', 'data/sdf-volumes/**/*.npy')

species = [row[10].replace('_', ' ') for row in dataset.rows]

taxons = []

csv_id = None

colors = dataset.get_colors()
colors = [np.array(color) for color in colors]
colors_by_node_id = [None for _ in range(len(nodes_by_id) + 1)]

for i, name in enumerate(species):
    current_taxons = tree.taxon_namespace.findall(name)
    if len(current_taxons) > 0:
        node = tree.find_node_for_taxon(current_taxons[0])
        colors_by_node_id[get_node_id(node)] = colors[i]
    if start_taxon.label == name and csv_id is None:
        csv_id = i

for node in tree.postorder_node_iter():
    if node.is_leaf():
        continue
    child_colors = [colors_by_node_id[get_node_id(child_node)] for child_node in node.child_nodes()]
    child_colors = [c for c in child_colors if c is not None]
    color = np.mean(np.stack(child_colors), axis=0)
    colors_by_node_id[get_node_id(node)] = color

stop_colors = [colors_by_node_id[node_id] for node_id in stop_ids]

from model.autoencoder import Autoencoder, LATENT_CODE_SIZE
autoencoder = Autoencoder(is_variational=False)
autoencoder.load()
autoencoder.eval()

stop_latent_codes = []

with torch.no_grad():
    voxels, meta = dataset[csv_id]

    if USE_VOLUME_NEURON:
        volume_data = [float(meta[4])]
    
    stop_latent_codes.append(autoencoder.encode(voxels.to(device), volume=volume_data if USE_VOLUME_NEURON else None))

stop_latent_codes += [latent_codes_by_node_id[node_id] for node_id in stop_ids[1:]]

viewer = MeshRenderer(size=1080, start_thread=False)
viewer.rotation = (130+180, 20)
viewer.model_color = (0.6, 0.6, 0.6)

font = ImageFont.truetype('helvetica.ttf', 60)

stop_index = 0
stop_count = len(stop_ids)
with torch.no_grad():
    for i in tqdm(range(FRAME_COUNT)):
        age = max_age * i / (FRAME_COUNT - 1)
        if stop_index < stop_count - 2 and stop_ages[stop_index + 1] < age:
            stop_index += 1
        progress = (age - stop_ages[stop_index]) / (stop_ages[stop_index + 1] - stop_ages[stop_index])
        
        viewer.model_color = stop_colors[stop_index] * (1.0 - progress) + stop_colors[stop_index + 1] * progress
        frame_latent_code = stop_latent_codes[stop_index] * (1.0 - progress) + stop_latent_codes[stop_index + 1] * progress
        viewer.set_voxels(autoencoder.decode(frame_latent_code))

        image = viewer.get_image(flip_red_blue=True)

        img = Image.fromarray(np.uint8(image))
        d = ImageDraw.Draw(img)
        d.text((50, 980), '{:0.0f}M years ago'.format(age), font=font, fill=(0, 0, 0))

        # headline
        if i <= 75:
            name = dataset.rows[csv_id][2]
            color = int(max(0, min(255, (i - 60) / 15 * 255)))
            width, _ = d.textsize(name, font=font)
            d.text((540 - width // 2, 50), name, font=font, fill=(color, color, color))

        image = np.array(img)[:, :, :3]

        cv2.imwrite("images/frame-{:05d}.png".format(i), image)

for i in range(FRAME_COUNT, FRAME_COUNT + STILL_FRAMES):
    cv2.imwrite("images/frame-{:05d}.png".format(i), image)

print("\n\nUse this command to create a video:\n")
print('ffmpeg -framerate 30 -i images/frame-%05d.png -c:v libx264 -profile:v high -crf 19 -pix_fmt yuv420p ancestor_animation_{:s}.mp4'.format(start_taxon.label.replace(' ', '_')))