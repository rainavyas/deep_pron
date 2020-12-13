import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import pickle
import sys
import os
import argparse
from pkl2pqvects_noI import get_vects, get_phones

# Get command line arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('PKL', type=str, help='Specify input pkl file')
commandLineParser.add_argument('OUT', type=str, help='Specify output pkl file')

args = commandLineParser.parse_args()
pkl_file = args.PKL
out_file = args.OUT

# Save the command run
if not os.path.isdir('CMDs'):
    os.mkdir('CMDs')
with open('CMDs/calculate_kl.cmd', 'a') as f:
    f.write(' '.join(sys.argv)+'\n')

pkl = pickle.load(open(pkl_file, "rb"))
print("Loaded pkl")

N = len(pkl['plp'])

# Get the batched tensors
X1, X2, M1, M2 = get_vects(pkl, phones, N, F=1000)

# Convert to tensors
X1 = torch.from_numpy(X1).float()
X2 = torch.from_numpy(X2).float()
M1 = torch.from_numpy(M1).float()
M2 = torch.from_numpy(M2).float()

# Get the output kl-distance values
y = calculate_kl(X1, X2, M1, M2)
print("Calculated KL distances")

# Save to pkl file
pkl_obj = y.tolist()
pickle.dump(pkl_obj, open(out_file, "wb"))
