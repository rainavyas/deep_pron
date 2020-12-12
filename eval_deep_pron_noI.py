import torch
import numpy as np
import pickle
import sys
import os
import argparse
from pkl2pqvects_noI import get_vects, get_phones
from model_deep_pron_noI import Deep_Pron
from utility import *

# Get command line arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('PKL', type=str, help='Specify input pkl file')
commandLineParser.add_argument('MODEL', type=str, help='Specify path of trained model')
commandLineParser.add_argument('--F', default=1000, type=int, help='Specify maximum number of frames in phone instance')

args = commandLineParser.parse_args()
pkl_file = args.PKL
model_file = args.MODEL
F = args.F

# Save the command run
if not os.path.isdir('CMDs'):
    os.mkdir('CMDs')
with open('CMDs/eval_deep_pron_noI.cmd', 'a') as f:
    f.write(' '.join(sys.argv)+'\n')

pkl = pickle.load(open(pkl_file, "rb"))
print("Loaded pkl")

# Get the phones
phones = get_phones()
N = len(pkl['pdf'])

# Get the batched tensors
X1, X2, M1, M2 = get_vects(pkl, phones, N, F)

# Get the output labels
y = (pkl['score'])

# Convert to tensors
X1 = torch.from_numpy(X1).float()
X2 = torch.from_numpy(X2).float()
M1 = torch.from_numpy(M1).float()
M2 = torch.from_numpy(M2).float()
y = torch.FloatTensor(y)
y = y[:N]

# Load the deep pron model
deep_model = torch.load(model_file)
deep_model.eval()

y_pred = deep_model(X1, X2, M1, M2)
y_pred[y_pred<0] = 0.0
y_pred[y_pred>6] = 6.0

# Calculate stats
mse = calculate_mse(y_pred.tolist(), y.tolist())
pcc = calculate_pcc(y_pred, y)
less05 = calculate_less05(y_pred, y)
less1 = calculate_less1(y_pred, y)
avg = torch.mean(y_pred)

print("STATs")
print("MSE: ", mse)
print("PCC: ", pcc)
print("Percentage less than 0.5 away: ", less05)
print("Percentage less than 1.0 away ", less1)
print("Average Predicted Grade ", avg)
