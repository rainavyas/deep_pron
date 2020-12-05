import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import pickle
import sys
import os
import argparse
from model_siamese import Siamese

# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA! But still using cpu")
        return torch.device('cpu')
    else:
        print("No CUDA found")
        return torch.device('cpu')

def LtoM(L, feature_dim, max_frame_len):
    # Make mask from lengths matrix
    v0 = [0]*feature_dim
    v1 = [1]*feature_dim
    M = []
    for l in L:
        sequence = [v1]*l + [v0]*(max_frame_len - l)
        M.append(sequence)
    return M

def raw2Tensor(X_raw, max_len, feature_dim):
    X = []
    v0 = [0]*feature_dim
    for sample in X_raw:
        l = len(sample)
        sequence = sample.tolist() + [v0]*(max_len-l)
        X.append(sequence)
    return X

# Get command line arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('PAIRS', default=1.0, type=str, help='Specify pkl file with siamese pairs data')

args = commandLineParser.parse_args()
pkl_file = args.PAIRS

# Save the command run
if not os.path.isdir('CMDs'):
    os.mkdir('CMDs')
with open('CMDs/train_siamese.cmd', 'a') as f:
    f.write(' '.join(sys.argv)+'\n')

# set the device
device = get_default_device()

# Define a max length of number of frames in a phone instance
max_frame_len = 100
mfcc_dim = 13

# Load the data
pkl = pickle.load(open(pkl_file, "rb"))
inds = [k for k in pkl if pkl[k][0].shape[0] < max_frame_len and pkl[k][1].shape[0] < max_frame_len]
X1_raw = [pkl[k][0] for k in inds]
X2_raw = [pkl[k][1] for k in inds]
y = [pkl[k][2] for k in inds]
lengths1 = [pkl[k][0].shape[0] for k in inds]
lengths2 = [pkl[k][1].shape[0] for k in inds]

# Make masks
M1 = LtoM(lengths1, mfcc_dim, max_frame_len)
M2 = LtoM(lengths2, mfcc_dim, max_frame_len)

# Make a fixed length tensor from variable length inputs
X1 = raw2Tensor(X1_raw, max_frame_len, mfcc_dim)
X2 = raw2Tensor(X2_raw, max_frame_len, mfcc_dim)

# Convert to tensors
X1 = torch.FloatTensor(X1)
X2 = torch.FloatTensor(X2)
M1 = torch.FloatTensor(M1)
M2 = torch.FloatTensor(M2)
y = torch.FloatTensor(y)

# Define training constants
lr = 5*1e-2
epochs = 20
bs = 10000
sch = 0.985
seed = 1
torch.manual_seed(seed)
'''
# Store all training dataset in a single wrapped tensor
train_ds = TensorDataset(X1, X2, M1, M2)

# Use DataLoader to handle minibatches easily
train_dl = DataLoader(train_ds, batch_size = bs, shuffle = True)

model = Siamese(mfcc_dim)
model.to(device)
print('model initialised')

criterion = torch.nn.MSELoss(reduction = 'mean')
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum = 0.9, nesterov=True)
# Scheduler for an adpative learning rate
# Every step size number of epochs, lr = lr * gamma
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = sch)

for epoch in range(epochs):
    model.train()
    for x1, x2, m1, m2 in train_dl:

        x1.unsqueeze(0)
        x2.unsqueeze(0)
        m1.unsqueeze(0)
        m2.unsqueeze(0)

        # Forward Pass
        d_pred = model(x1, x2, m1, m2)

        # Compute loss
        log_d_pred = torch.log(d_pred.squeeze())
        diff = log_d_pred - y
'''
