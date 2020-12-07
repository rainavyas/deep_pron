import torch

class Deep_Pron(torch.nn.Module):
    def __init__(self, num_features=13):
        super(Siamese, self).__init__()
        self.lstm = torch.nn.LSTM(input_size = num_features, hidden_size=num_features, num_layers=1, batch_first=True, bidirectional=True)
        self.attn = torch.nn.Linear(2*num_features, 2*num_features, bias = False)
        self.attn2 = torch.nn.Linear(2*num_features, 2*num_features, bias = False)

        # Parameters for FCC
        self.bn1 = torch.nn.BatchNorm1d(num_features=num_features)
        self.fc1 = torch.nn.Linear(num_features, 1000)
        self.fc2 = torch.nn.Linear(1000, 1000)
        self.fc3 = torch.nn.Linear(1000, 1000)
        self.fc4 = torch.nn.Linear(1000, 1000)
        self.fc5 = torch.nn.Linear(1000, 1000)
        self.fc6 = torch.nn.Linear(1000, 1000)
        self.fc7 = torch.nn.Linear(1000, 1)
        self.drop_layer = torch.nn.Dropout(p=0.5)

    def apply_LSTM(self, X, M):
        '''
        X = input matrix
        M = mask
        '''

        # Create a lengths matrix
        lens = torch.sum(M[:,:,:,:,0], dim = -1)

        # Flatten lens and X to use in nn.lstm module
        Xf = X.view(X.size(0)*X.size(1)*X.size(2), X.size(3), X.size(4))
        lensf = lens.view(X.size(0)*X.size(1)*X.size(2))

        # Pack padded tensor into a padded sequence object
        Xp = torch.nn.utils.rnn.pack_padded_sequence(Xf, lensf, batch_first = True, enforce_sorted = False)

        # Pass through LSTM
        outputs, hidden_states = self.lstm(Xp)

        # Convert packed object to padded tensor
        O = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first = True, total_length = X.size(2))
        Op = O[0]

        # Unflatten the tensor
        X_after_RNN = torch.reshape(Op, (X.size(0), X.size(1), X.size(2), X.size(3), -1))

        return X_after_RNN

    def apply_attention_over_frames(self, X, A, M):
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
        S_half = torch.einsum('buvwi,ij->buvwj', X, A)
        S = torch.einsum('buvwi,buvwi->buvw', X, S_half)
        T = torch.nn.Tanh()
        ST = T(S)
        # Use mask to convert padding scores to -inf (go to zero after softmax normalisation)
        ST_masked = ST + M_useful[:,:,:,:,0]
        # Normalise weights using softmax for each utterance of each speaker
        SM = torch.nn.Softmax(dim = 3)
        W = SM(ST_masked)
        # Perform weighted sum (using normalised scores above) along the words axes for X
        weights_extra_axis = torch.unsqueeze(W, 4)
        repeated_weights = weights_extra_axis.expand(-1, -1, -1, -1, X.size(4))
        x_multiplied = X * repeated_weights
        x_attn = torch.sum(x_multiplied, dim = 3)

        return x_attn

    def apply_attention_over_instances(self, X, A, M):
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

    def forward(X1, X2, M1, M2):
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

        # Pass frames per instance through LSTM
        X1_after_LSTM = self.apply_LSTM(X1, M1)
        X2_after_LSTM = self.apply_LSTM(X2, M2)

        # Apply attention over frames per instance
        A = self.attn(torch.eye(X1_after_LSTM.size(-1)))
        X1_after_frame_attn = self.apply_attention_over_frames(X1_after_LSTM, A, M1)
        X2_after_frame_attn = self.apply_attention_over_frames(X2_after_LSTM, A, M2)

        # Apply attention over instances per phones
        A2 = self.attn2(torch.eye(X1_after_frame_attn.size(-1)))
        X1_after_inst_attn = self.apply_attention_over_frames(X1_after_frame_attn, A, M1)
        X2_after_inst_attn = self.apply_attention_over_frames(X2_after_frame_attn, A, M2)

        # Calculate phone distances using l2-norm
        d1 = (X1-X2)**2
        d = torch.sum(d1, dim=-1)

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
