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


def calculate_kl(p_vects, q_vects, p_frames_mask, q_frames_mask):
    # Get p/q_lengths
    p_lengths = torch.sum(p_frames_mask[:,:,:,0].squeeze(), dim=2).unsqueeze(dim=2).repeat(1,1,13)
    q_lengths = torch.sum(q_frames_mask[:,:,:,0].squeeze(), dim=2).unsqueeze(dim=2).repeat(1,1,13)

    p_lengths[p_lengths==0] = 1
    q_lengths[q_lengths==0] = 1

    # Compute means
    p_means = torch.sum(p_vects, dim=2)/p_lengths
    q_means = torch.sum(q_vects, dim=2)/q_lengths

    # Compute the p/q_covariances tensor
    p_vects_unsq = torch.unsqueeze(p_vects, dim=4)
    q_vects_unsq = torch.unsqueeze(q_vects, dim=4)

    p_vects_unsq_T = torch.transpose(p_vects_unsq, 3, 4)
    q_vects_unsq_T = torch.transpose(q_vects_unsq, 3, 4)

    p_means_squared = torch.squeeze(torch.sum(torch.matmul(p_vects_unsq, p_vects_unsq_T), dim=2)/p_lengths.unsqueeze(dim=3).repeat(1,1,1,13))
    q_means_squared = torch.squeeze(torch.sum(torch.matmul(q_vects_unsq, q_vects_unsq_T), dim=2)/q_lengths.unsqueeze(dim=3).repeat(1,1,1,13))

    p_means_unsq = torch.unsqueeze(p_means, dim=3)
    q_means_unsq = torch.unsqueeze(q_means, dim=3)

    p_means_unsq_T = torch.transpose(p_means_unsq, 2, 3)
    q_means_unsq_T = torch.transpose(q_means_unsq, 2, 3)

    p_m2 = torch.squeeze(torch.matmul(p_means_unsq, p_means_unsq_T))
    q_m2 = torch.squeeze(torch.matmul(q_means_unsq, q_means_unsq_T))

    p_covariances = p_means_squared - p_m2
    q_covariances = q_means_squared - q_m2

    # If no phone, make covariance matrix identity
    # Need to first calculate num_phones_mask
    num_phones_mask_p = p_frames_mask[:,:,0,0].squeeze()
    num_phones_mask_q = q_frames_mask[:,:,0,0].squeeze()
    num_phones_mask = num_phones_mask_p + num_phones_mask_q
    num_phones_mask[num_phones_mask==1]=0
    num_phones_mask[num_phones_mask==2]=1

    p_covariances_shifted = p_covariances - torch.eye(13)
    q_covariances_shifted = q_covariances - torch.eye(13)

    p_covariances_shifted_masked = p_covariances_shifted * num_phones_mask.unsqueeze(dim=2).repeat(1,1,13).unsqueeze(dim=3).repeat(1,1,1,13)
    q_covariances_shifted_masked = q_covariances_shifted * num_phones_mask.unsqueeze(dim=2).repeat(1,1,13).unsqueeze(dim=3).repeat(1,1,1,13)

    p_covs = p_covariances_shifted_masked + torch.eye(13)
    q_covs = q_covariances_shifted_masked + torch.eye(13)

    # Calculate the symmetric KL divergences
    p = torch.distributions.MultivariateNormal(p_means, p_covs)
    q = torch.distributions.MultivariateNormal(q_means, q_covs)

    kl_loss = ((torch.distributions.kl_divergence(p, q) + torch.distributions.kl_divergence(q, p))*0.5)

    return kl_loss

# Get command line arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('PKL', type=str, help='Specify input pkl file')
commandLineParser.add_argument('OUT', type=str, help='Specify output pt file')
commandLineParser.add_argument('--N', default=993, type=int, help='Specify number of speakers')
commandLineParser.add_argument('--F', default=1000, type=int, help='Specify maximum number of frames in phone instance')

args = commandLineParser.parse_args()
pkl_file = args.PKL
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
y = calculate_kl(X1, X2, M1, M2)
print("Calculated KL distances")

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
y_train = y[validation_size:N]
y_val = y[:validation_size]

# Define training constants
lr = 8*1e-2
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
torch.save(deep_model, out_file)
