import torch
import torch.nn.functional as F


class FCC(torch.nn.Module):
    def __init__(self, num_features):

        super(FCC, self).__init__()
        self.bn1 = torch.nn.BatchNorm1d(num_features=num_features)
        self.fc1 = torch.nn.Linear(num_features, 1000)
        self.fc2 = torch.nn.Linear(1000, 1000)
        self.fc3 = torch.nn.Linear(1000, 1000)
        self.fc4 = torch.nn.Linear(1000, 1000)
        self.fc5 = torch.nn.Linear(1000, 1000)
        self.fc6 = torch.nn.Linear(1000, 1000)
        self.fc7 = torch.nn.Linear(1000, 1)
        self.drop_layer = torch.nn.Dropout(p=0.5)

    def forward(self, p_means, p_covariances, q_means, q_covariances, num_phones_mask):
        '''
        p/q_means = [num_speakers X num_feats X mfcc_dim]
        p/q_covariances = [num_speakers X num_feats X mfcc_dim X mfcc_dim]
        num_phones_mask = [num_speakers X num_feats],
        with a 0 corresponding to positiion that should be -1 (no phones observed)
        and a 1 everywhere else.
        n.b. num_feats = 46*47*0.5 = 1128 usually
        '''

        # compute symmetric kl-divergences between every phone distribution per speaker
        p = torch.distributions.MultivariateNormal(p_means, p_covariances)
        q = torch.distributions.MultivariateNormal(q_means, q_covariances)

        kl_loss = ((torch.distributions.kl_divergence(p, q) + torch.distributions.kl_divergence(q, p))*0.5)

        # log all the features
        # add small error to mak 0-kl distances not a NaN
        X = kl_loss + (1e-5)
        feats = torch.log(X)

        # Apply mask to get -1 features in correct place (i.e. where no phones observed)
        feats_shifted = feats + 1
        feats_masked = feats_shifted * num_phones_mask
        feats_correct = feats_masked - 1

        # pass through layers

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
