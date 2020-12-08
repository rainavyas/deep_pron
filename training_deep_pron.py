import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import pickle
import sys
import os
import argparse
from pkl2pqvects import get_vects, get_phones
from model_deep_pron import Deep_Pron
from model_siamese import Siamese


# Get command line arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('PKL', type=str, help='Specify input pkl file')
commandLineParser.add_argument('OUT', type=str, help='Specify output pt file')
commandLineParser.add_argument('SIAM', type=str, help='Specify pre-trained Siamese .pt model')
commandLineParser.add_argument('--F', default=100, type=int, help='Specify maximum number of frames in phone instance')
commandLineParser.add_argument('--I', default=500, type=int, help='Specify maximum number of instances of a phone')

args = commandLineParser.parse_args()
pkl_file = args.PKL
out_file = args.OUT
siam_model = args.SIAM
F = args.F
I = args.I

# Save the command run
if not os.path.isdir('CMDs'):
    os.mkdir('CMDs')
with open('CMDs/training_deep_pron.cmd', 'a') as f:
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

# Split into training and validation sets
validation_size = 100
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
# Define training constants
lr = 8*1e-2
epochs = 20
bs = 100
sch = 0.985
seed = 1
torch.manual_seed(seed)

# Store all training dataset in a single wrapped tensor
train_ds = TensorDataset(X1_train, X2_train, M1_train, M2_train, y_train)

# Use DataLoader to handle minibatches easily
train_dl = DataLoader(train_ds, batch_size = bs, shuffle = True)

# Load the pre-trained siamese model
siamese_model = torch.load(siam_model)

# Initialise deep pron model to be trained
deep_model = Deep_Pron()
#deep_model.to(device)

# Transfer requried learnt parameters of pre-trained Siamese model
siamese_model_dict = siamese_model.state_dict()
deep_model_dict = deep_model.state_dict()
model_dict = deep_model_dict
# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in siamese_model_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
# 3. load the new state dict
deep_model.load_state_dict(model_dict)
print("Initialised model")

criterion = torch.nn.MSELoss(reduce='mean')
optimizer = torch.optim.SGD(deep_model.parameters(), lr=lr, momentum = 0.9, nesterov=True)
# Scheduler for an adpative learning rate
# Every step size number of epochs, lr = lr * gamma
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = sch)

for epoch in range(epochs):
    deep_model.train()
    print("On Epoch, ", epoch)

    for x1, x2, m1, m2, yb in train_dl:

        # Forward pass
        y_pred = deep_model(x1, x2, m1, m2)

        # Compute loss
        loss = criterion(y_pred, yb)

        # Zero gradients, backward pass, update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("running loss: ", loss.item())

    # Validation
    deep_model.eval()
    mse_loss = deep_model(X1_val, X2_val, M1_val, M2_val)
    print("Validation Loss: ", mse_loss.item())

# Save the trained model
torch.save(deep_model, out_file)
