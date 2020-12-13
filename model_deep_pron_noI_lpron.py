import torch
import torch.nn.functional as F

class Deep_Pron(torch.nn.Module):
    def __init__(self, num_features=13):
        super(Deep_Pron, self).__init__()
        num_distance_features = 1128
        self.bn2D = torch.nn.BatchNorm2d(num_features=num_distance_features)
        self.attn = torch.nn.Linear(num_features, num_features, bias = False)

        # Parameters for FCC
        self.bn1 = torch.nn.BatchNorm1d(num_features=num_distance_features)
        self.fc1 = torch.nn.Linear(num_distance_features, 1000)
        self.fc2 = torch.nn.Linear(1000, 1000)
        self.fc3 = torch.nn.Linear(1000, 1000)
        self.fc4 = torch.nn.Linear(1000, 1000)
        self.fc5 = torch.nn.Linear(1000, 1000)
        self.fc6 = torch.nn.Linear(1000, 1000)
        self.fc7 = torch.nn.Linear(1000, 1)
        self.drop_layer = torch.nn.Dropout(p=0.5)

    def apply_attention(self, X, A, M):
        '''
        X = input
        A = Weights matrix to be learnt
        M = Mask

        Slightly different usual attention,
        usually alpha_i = f(query_j, key_i)
        and h_j = \sum_i{alpha_i*value_i}

        This implementation:
        No query,
        alpha_i = f(key_i, value_i)
        key = value
        h = \sum_i{alpha_i*value_i}
        No repeat over j, as no query - i.e. single vector output
        '''

        # Make mask useful, so that 0 in useful positions and -inf elsewhere
        M_useful = (M - 1)*100000
        S_half = torch.einsum('buvi,ij->buvj', X, A)
        S = torch.einsum('buvi,buvi->buv', X, S_half)
        T = torch.nn.Tanh()
        ST = T(S)
        # Use mask to convert padding scores to -inf (go to zero after softmax normalisation)
        ST_masked = ST + M_useful[:,:,:,0]
        # Normalise weights using softmax for each utterance of each speaker
        SM = torch.nn.Softmax(dim = 2)
        W = SM(ST_masked)
        # Perform weighted sum (using normalised scores above) along the words axes for X
        weights_extra_axis = torch.unsqueeze(W, 3)
        repeated_weights = weights_extra_axis.expand(-1, -1, -1, X.size(3))
        x_multiplied = X * repeated_weights
        x_attn = torch.sum(x_multiplied, dim = 2)

        return x_attn

    def forward(self, X1, X2, M1, M2):
        '''
        Xi = [N X P*(P-1)*0.5 x I x F X n]
        N = batch size (number of speakers)
        P = phones (e.g. 47)
        I = Max number of phone instances per phone
        F = Max number of frames per instance
        n = Number of features -> usually it is MFCC dimension = 13

        Mi = mask with same dimensions as X, with 1s in positions of value
        and 0 elsewhere
        '''
        # Apply batch normalisation
        batched_X1 = self.bn2D(X1)
        batched_X2 = self.bn2D(X2)

        # Apply attention over frames
        A = self.attn(torch.eye(batched_X1.size(-1)))
        X1_after_frame_attn = self.apply_attention(batched_X1, A, M1)
        X2_after_frame_attn = self.apply_attention(batched_X2, A, M2)

        # Calculate phone distances using l2-norm
        d1 = (X1_after_frame_attn-X2_after_frame_attn)**2
        d = torch.sum(d1, dim=-1)

        # log all the features
        # add small error to mak 0-kl distances not a NaN
        X = d + (1e-5)
        feats = torch.log(X)

        # Make zero phone observation positions zero
        # Need to first calculate num_phones_mask
        num_phones_mask_p = M1[:,:,0,0].squeeze()
        num_phones_mask_q = M2[:,:,0,0].squeeze()
        num_phones_mask = num_phones_mask_p + num_phones_mask_q
        num_phones_mask[num_phones_mask==1]=0
        num_phones_mask[num_phones_mask==2]=1

        # Apply mask to get -1 features in correct place (i.e. where no phones observed)
        feats_shifted = feats + 1
        feats_masked = feats_shifted * num_phones_mask
        feats_correct = feats_masked - 1

        # Pass through FCC layers
        # Normalize each input vector
        X_norm = self.bn1(feats_correct)
        h1 = F.relu(self.fc1(X_norm))
        h2 = F.relu(self.fc2(self.drop_layer(h1)))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(self.drop_layer(h3)))
        h5 = F.relu(self.fc5(h4))
        h6 = F.relu(self.fc6(h5))
        y = self.fc7(h6)

        return y.squeeze()
