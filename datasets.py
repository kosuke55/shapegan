import torch
from torch.utils.data import Dataset
import os
import numpy as np
import random

class VoxelDataset(Dataset):
    def __init__(self, files, clamp=10, rescale_sdf=True):
        self.files = files
        self.clamp = clamp
        self.rescale_sdf = rescale_sdf
        self.remove_nan = True

    def __len__(self):
        return len(self.files)

    def get_file_name(self, index):
        return os.path.basename(self.files[index])

    def __getitem__(self, index):
        array = np.load(self.files[index])

        if array.dtype == np.float64:
            array = array.astype(np.float32)

        file_name = self.get_file_name(index)
        if file_name.startswith('baboon'):
            array = array.transpose([0, 2, 1])
        elif file_name.startswith('human'):
            array = np.array(array.transpose([0, 2, 1]))
        else:
            array = np.array(array.transpose([0, 2, 1])[::-1, ::-1, ::-1])

        if self.remove_nan:
            array[~np.isfinite(array)] = 0

        result = torch.from_numpy(array)
        if self.clamp is not None:
            result.clamp_(-self.clamp, self.clamp)
            if self.rescale_sdf:
                result /= self.clamp
        return (result, [])

    @staticmethod
    def glob(pattern):
        import glob
        files = glob.glob(pattern, recursive=True)
        if len(files) == 0:
            raise Exception('No files found for glob pattern {:s}.'.format(pattern))
        return VoxelDataset(sorted(files))

    def show(self):
        from rendering import MeshRenderer
        import time
        from tqdm import tqdm

        viewer = MeshRenderer()
        for item in tqdm(self):
            voxels, _ = item
            viewer.set_voxels(voxels.numpy())
            time.sleep(0.5)

# Voxel dataset of individual .npy files that uses a CSV file to get metadata for the specimen
class CSVVoxelDataset(VoxelDataset):
    def __init__(self, csv_file_name, voxel_file_glob_pattern):
        import csv
        with open(csv_file_name, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            csv_iterator = iter(csv_reader)
            next(csv_iterator)
            csv_rows = list(csv_iterator)

        import glob
        file_names = glob.glob(voxel_file_glob_pattern, recursive=True)

        used_file_names = []
        used_rows = []

        for row in csv_rows:
            found_matching_file = False
            for file_name in file_names:
                if os.path.basename(file_name) == row[0] + '.npy':
                    used_rows.append(row + [file_name])
                    used_file_names.append(file_name)
                    found_matching_file = True
                    break
            if not found_matching_file:
                print('Warning: Could not find a .npy for the specimen ID "{:s}". Skipping this line.'.format(row[0]))
        
        if len(used_rows) == 0:
            raise ValueError('Found no .npy files with matching specimen IDs in the CSV file (out of {:d} rows in the CSV file and {:d} .npy files found.'.format(len(csv_rows), len(file_names)))

        self.rows = used_rows

        VoxelDataset.__init__(self, used_file_names)

    def __getitem__(self, index):
        voxels, _ = VoxelDataset.__getitem__(self, index)
        return (voxels, self.rows[index])
    
    def show(self):
        from rendering import MeshRenderer
        import time
        from tqdm import tqdm
        import pygame

        viewer = MeshRenderer()
        for item in tqdm(self):
            voxels, meta = item
            pygame.display.set_caption(meta[2])
            color = meta[1]
            viewer.set_voxels(voxels.numpy())
            viewer.model_color = (int(color[3:5], 16) / 255, int(color[5:7], 16) / 255, int(color[7:], 16) / 255)
            tqdm.write('Showing a {:s} from file "{:s}"'.format(meta[2], meta[-1]))
            time.sleep(0.5)

    def get_row(self, index):
        return self.rows[index]

# This dataset is balanced so that samples from each category are used at the same frequency.
# The .shuffle() method should be called after each epoch.
class BalancedDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

        self.indices_by_category = {}

        def get_category(index):
            return base_dataset.rows[index][2] # DisplayName

        for index in range(len(base_dataset)):
            category = get_category(index)
            if category not in self.indices_by_category:
                self.indices_by_category[category] = []
            self.indices_by_category[category].append(index)

        self.number_of_items_per_category = len(base_dataset) // len(self.indices_by_category.keys())

        self.shuffle()
    
    def shuffle(self):
        indices = []
        for category in self.indices_by_category.keys():
            category_indices = self.indices_by_category[category]

            for _ in range(self.number_of_items_per_category // len(category_indices)):
                indices.extend(category_indices)

            indices.extend(random.sample(category_indices, self.number_of_items_per_category % len(category_indices)))

        random.shuffle(indices)
        self.indices = indices

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        return self.base_dataset[self.indices[index]]

    def get_row(self, index):
        return self.base_dataset.get_row(self.indices[index])


if __name__ == '__main__':
    #dataset = VoxelDataset.glob('data/sdf-volumes/**/*.npy')
    dataset = CSVVoxelDataset('data/color-name-volume-mapping-bc-primates.csv', 'data/sdf-volumes/**/*.npy')
    
    dataset.show()