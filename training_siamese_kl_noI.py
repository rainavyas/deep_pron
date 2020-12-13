import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import pickle
import sys
import os
import argparse
from pkl2pqvects_noI import get_vects, get_phones
from model_siamese_noI import Siamese
from utility import calculate_mse

# Get command line arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('PKL', type=str, help='Specify input pkl file')
commandLineParser.add_argument('KL', type=str, help='Specify labels pkl file with kl distances')
commandLineParser.add_argument('OUT', type=str, help='Specify output pt file')
commandLineParser.add_argument('--N', default=993, type=int, help='Specify number of speakers')
commandLineParser.add_argument('--F', default=1000, type=int, help='Specify maximum number of frames in phone instance')

args = commandLineParser.parse_args()
pkl_file = args.PKL
kl_file = args.KL
out_file = args.OUT
N = args.N
F = args.F

# Save the command run
if not os.path.isdir('CMDs'):
    os.mkdir('CMDs')
with open('CMDs/training_siamese_kl_noI.cmd', 'a') as f:
    f.write(' '.join(sys.argv)+'\n')

pkl = pickle.load(open(pkl_file, "rb"))
print("Loaded pkl")

# Get the phones
phones = get_phones()

# Get the batched tensors
X1, X2, M1, M2 = get_vects(pkl, phones, N, F)

# Convert to tensors
X1 = torch.from_numpy(X1).float()
X2 = torch.from_numpy(X2).float()
M1 = torch.from_numpy(M1).float()
M2 = torch.from_numpy(M2).float()

# Get the output kl-distance values
y = pickle.load(open(kl_file, "rb"))
y = torch.FloatTensor(y)
y = y[:N]
print("Got KL distances")

# Split into training and validation sets
validation_size = 50
X1_train = X1[validation_size:]
X1_val = X1[:validation_size]
X2_train = X2[validation_size:]
X2_val = X2[:validation_size]
M1_train = M1[validation_size:]
M1_val = M1[:validation_size]
M2_train = M2[validation_size:]
M2_val = M2[:validation_size]
y_train = y[validation_size:]
y_val = y[:validation_size]

# Define training constants
lr = 8*1e-11
epochs = 20
bs = 50
sch = 0.985
seed = 1
torch.manual_seed(seed)

# Store all training dataset in a single wrapped tensor
train_ds = TensorDataset(X1_train, X2_train, M1_train, M2_train, y_train)

# Use DataLoader to handle minibatches easily
train_dl = DataLoader(train_ds, batch_size = bs, shuffle = True)

# Initialise the Siamese Network to be trained
siamese_model = Siamese()
print("Initialised Model")

criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(siamese_model.parameters(), lr=lr, momentum = 0.9, nesterov=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = sch)

for epoch in range(epochs):
    siamese_model.train()
    print("On Epoch, ", epoch)

    for x1, x2, m1, m2, yb in train_dl:

        # Forward pass
        y_pred = siamese_model(x1, x2, m1, m2)

        # Compute loss
        y_pred_flat = torch.reshape(y_pred, (-1,)).squeeze()
        yb_flat = torch.reshape(yb, (-1,)).squeeze()
        loss = criterion(y_pred, yb)

        # Zero gradients, backward pass, update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("running loss: ", loss.item())

    # Validation
    siamese_model.eval()
    y_val_pred = siamese_model(X1_val, X2_val, M1_val, M2_val)
    y_val_pred_flat = torch.reshape(y_val_pred, (-1,)).squeeze()
    y_val_flat = torch.reshape(y_val, (-1,)).squeeze()
    mse_loss = calculate_mse(y_val_pred_flat, y_val_flat)
    print("Validation Loss: ", mse_loss)

# Save the trained model
torch.save(siamese_model, out_file)
