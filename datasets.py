import torch
from torch.utils.data import Dataset
import os
import numpy as np

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
        return result

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
            viewer.set_voxels(item.numpy())
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
            for file_name in file_names:
                if os.path.basename(file_name) == row[0] + '.npy':
                    used_rows.append(row)
                    used_file_names.append(file_name)
                    break
        
        if len(used_rows) == 0:
            raise ValueError('Found no .npy files with matching specimen IDs in the CSV file (out of {:d} rows in the CSV file and {:d} .npy files found.'.format(len(rows), len(file_names)))

        self.rows = used_rows

        VoxelDataset.__init__(self, used_file_names)


if __name__ == '__main__':
    #dataset = VoxelDataset.glob('data/sdf-volumes/**/*.npy')
    dataset = CSVVoxelDataset('data/color-name-volume-mapping-bc-primates.csv', 'data/sdf-volumes/**/*.npy')
    
    dataset.show()