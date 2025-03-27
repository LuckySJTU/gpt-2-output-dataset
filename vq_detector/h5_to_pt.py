import torch
import numpy as np
from tqdm import tqdm
import h5py
import sys

args = sys.argv
assert len(args) == 3, 'Please identify data path and save path'
data_path = args[1]
save_path = args[2]

h5 = h5py.File(data_path)
x = np.zeros(h5['hidden_states'].shape)
for i in tqdm(range(x.shape[0])):
    x[i] = h5['hidden_states'][i]
x = torch.tensor(x)
torch.save(x, save_path)