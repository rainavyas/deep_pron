import torch

class Siamese(torch.nn.Module):
    def __init__(self, num_features=13):

        super(Siamese, self).__init__()
        num_distance_features = 1128
        self.attn = torch.nn.Linear(num_features, num_features, bias = False)
        self.layer = torch.nn.Linear(num_distance_features, num_distance_features)

    def apply_selfattn_no_query(self, X, A, M):
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
        Xi = [N x M x L x n]
        N = batch size (number of speakers)
        M = Samples (typically 100)
        L = Maximum Sequence Length of frames - padding in this dimension
        n = Number of features -> usually it is MFCC dimension = 13

        Mi = mask with same dimensions as X, with 1s in positions of value
        and 0 elsewhere

        X1 and X2 projected to a space H1 and H2, such that
        ||H1-H2|| conceptually represents phone distance d,
        ensured by training using the true KL-divergence between Gaussian
        distribution per phone per speaker
        '''

        #Apply attention over frames
        A = self.attn(torch.eye(X1.size(3)))
        H1 = self.apply_selfattn_no_query(X1, A, M1)
        H2 = self.apply_selfattn_no_query(X2, A, M2)

        # l2-norm to find the distance between sample pairs
        d1 = (H1-H2)**2
        d = torch.sum(d1, dim=-1)

        # Apply layer
        d_scaled = self.layer(d)

        return d_scaled
