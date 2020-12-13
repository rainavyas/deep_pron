import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import pickle
import sys
import os
import argparse
import math
from pkl2pqvects_noI import get_vects, get_phones

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

# Get the phones
phones = get_phones()

# Get the batched tensors
X1, X2, M1, M2 = get_vects(pkl, phones, N, F=1000)

# Convert to tensors
X1 = torch.from_numpy(X1).float()
X2 = torch.from_numpy(X2).float()
M1 = torch.from_numpy(M1).float()
M2 = torch.from_numpy(M2).float()

# Get the output kl-distance values in batches
y = []
bs = 20
for i in range(math.floor(N/bs)):
    print("On batch ", i)
    start = i*bs
    end = (i+1)*bs
    X1_curr = X1[start:end]
    X2_curr = X2[start:end]
    M1_curr = M1[start:end]
    M2_curr = M2[start:end]
    y_curr = calculate_kl(X1_curr, X2_curr, M1_curr, M2_curr)
    y.append(y_curr)
y = torch.cat(y, dim=0)
print("Calculated KL distances")

# Save to pkl file
pkl_obj = y.tolist()
pickle.dump(pkl_obj, open(out_file, "wb"))
