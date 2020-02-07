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
            raise Exception(
                'No files found for glob pattern {:s}.'.format(pattern))
        return VoxelDataset(sorted(files))

    def show(self):
        from rendering import MeshRenderer
        import time
        from tqdm import tqdm

        viewer = MeshRenderer()
        for item in tqdm(self):
            viewer.set_voxels(item.numpy())
            time.sleep(0.5)


if __name__ == '__main__':
    dataset = VoxelDataset.glob('data/sdf-volumes/**/*.npy')
    dataset.show()