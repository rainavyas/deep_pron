import torch
import numpy as np
import pickle
import sys
import os
import argparse
from pkl2pqvects import get_vects, get_phones
from model_deep_pron import Deep_Pron
from utility import *

# Get command line arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('PKL', type=str, help='Specify input pkl file')
commandLineParser.add_argument('MODEL', type=str, help='Specify path of trained model')
commandLineParser.add_argument('--F', default=100, type=int, help='Specify maximum number of frames in phone instance')
commandLineParser.add_argument('--I', default=500, type=int, help='Specify maximum number of instances of a phone')

args = commandLineParser.parse_args()
pkl_file = args.PKL
model_file = args.MODEL
F = args.F
I = args.I

# Save the command run
if not os.path.isdir('CMDs'):
    os.mkdir('CMDs')
with open('CMDs/eval_deep_pron.cmd', 'a') as f:
    f.write(' '.join(sys.argv)+'\n')

pkl = pickle.load(open(pkl_file, "rb"))
print("Loaded pkl")

# Get the phones
phones = get_phones()

# Get the batched tensors
X1, X2, M1, M2 = get_vects(pkl, phones, F, I)

# Get the output labels
y = (pkl['score'])

# Convert to tensors
X1 = torch.FloatTensor(X1.tolist())
X2 = torch.FloatTensor(X2.tolist())
M1 = torch.FloatTensor(M1.tolist())
M2 = torch.FloatTensor(M2.tolist())
y = torch.FloatTensor(y)

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
