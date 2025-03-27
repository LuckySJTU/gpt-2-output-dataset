import math
import os
import h5py
from torch.utils.data import Dataset, DataLoader
import torch
from constants import KEY_LM_HIDDEN_STATES, KEY_LM_INPUT_IDS, KEY_LM_LABELS
from utils import load_config
from tqdm import tqdm


class HDF5Dataset(Dataset):
    def __init__(self, h5_files_dir, split, batch_size):
        """
        Args:
            h5_files_dir (str): 
            split (str): train/valid/test
        """
        h5_file_path = os.path.join(h5_files_dir, split+'.h5')
        self.h5_file = h5py.File(h5_file_path, 'r')
        
        self.hidden_states = self.h5_file[KEY_LM_HIDDEN_STATES]
        self.batch_size = batch_size
        self.total_samples = self.hidden_states.shape[0]
        self.len = math.ceil(self.total_samples / self.batch_size)

        self.idxs = torch.arange(0, self.total_samples, self.batch_size)
        self.idxs = torch.cat([self.idxs, torch.tensor([self.total_samples])])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # return torch.tensor(self.hidden_states[idx], dtype=torch.float)
        return self.hidden_states[self.idxs[idx]:self.idxs[idx+1]]

    def close(self):
        self.h5_file.close()


# class ChunkedHDF5Dataset(HDF5Dataset):
#     def __init__(self, h5_files_dir, split, chunk_size:int):
#         super().__init__(h5_files_dir, split)
#         self.chunk_size = chunk_size
    
#     def __getitem__(self, idx):
#         item = super().__getitem__(idx)
#         hidden_states = item['hidden_states']
#         item['hidden_states'] = hidden_states[self.chunk_size-1::self.chunk_size, :]
#         return item


def get_chunked_h5dataloader(config_path, split, shuffle=None):
    config = load_config(config_path=config_path)
    num_workers = 16  # Set num workers to 0 to enable debugging
    if shuffle is None:
        shuffle = split == 'train'
    dataset = HDF5Dataset(config['h5_file_path'], split, batch_size=config['batch_size'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers)
    return dataloader


if __name__ == '__main__':
    import time
    start_time = time.time()
    dataloader = get_chunked_h5dataloader('conf/example.yaml', 'train')

    for batch in tqdm(dataloader):
        pass
        # hidden_states = batch
        # print(f"Hidden States: {hidden_states.shape}")
        # break  # 这里只打印一个批次的数据
    print(batch.shape)
    end_time = time.time()
    print(end_time - start_time)