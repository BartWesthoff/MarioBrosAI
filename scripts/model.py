import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ------------------------------
# Spectral Norm Residual Block
# ------------------------------
class SpectralBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.relu = nn.ReLU()
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + self.skip(x))

# ------------------------------
# Impala CNN Feature Extractor
# ------------------------------
class ImpalaCNN(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.MaxPool2d(3, stride=2),
            SpectralBlock(32, 32),
            SpectralBlock(32, 32)
        )
        self.block2 = nn.Sequential(
            SpectralBlock(32, 64),
            SpectralBlock(64, 64)
        )
        self.block3 = nn.Sequential(
            SpectralBlock(64, 64),
            SpectralBlock(64, 64)
        )
        self.pool = nn.AdaptiveMaxPool2d((6, 6))

    def forward(self, x):
        x = x / 255.0  # Normalize input
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        return x  # shape: [B, 64, 6, 6]

# ------------------------------
# IQN Embedding Layer
# ------------------------------
class IQNEmbedding(nn.Module):
    def __init__(self, n_cos=64, embedding_dim=64*6*6):
        super().__init__()
        self.n_cos = n_cos
        self.linear = nn.Linear(n_cos, embedding_dim)

    def forward(self, taus):
        batch_size, n_quantiles = taus.shape
        i = torch.arange(1, self.n_cos + 1, device=taus.device).float().view(1, 1, -1)
        cosines = torch.cos(i * math.pi * taus.unsqueeze(-1))
        out = F.relu(self.linear(cosines.view(batch_size * n_quantiles, -1)))
        return out.view(batch_size, n_quantiles, -1)

# ------------------------------
# Noisy Linear Layer
# ------------------------------
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.mu_w = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_w = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.mu_b = nn.Parameter(torch.empty(out_features))
        self.sigma_b = nn.Parameter(torch.full((out_features,), sigma_init))
        self.reset_parameters()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.mu_w.size(1))
        self.mu_w.data.uniform_(-mu_range, mu_range)
        self.mu_b.data.uniform_(-mu_range, mu_range)

    def forward(self, x):
        if self.training:
            noise_w = torch.randn_like(self.mu_w)
            noise_b = torch.randn_like(self.mu_b)
            return F.linear(x, self.mu_w + self.sigma_w * noise_w, self.mu_b + self.sigma_b * noise_b)
        else:
            return F.linear(x, self.mu_w, self.mu_b)

# ------------------------------
# Full BTR Network
# ------------------------------
class BTRNetwork(nn.Module):
    def __init__(self, num_actions, n_quantiles=8, n_cos=64):
        super().__init__()
        self.feature_extractor = ImpalaCNN()
        self.quantile_embedding = IQNEmbedding(n_cos)
        self.n_quantiles = n_quantiles

        fc_input_dim = 64 * 6 * 6

        # Value Stream
        self.value_stream = nn.Sequential(
            NoisyLinear(fc_input_dim, 512),
            nn.ReLU(),
            NoisyLinear(512, 1)
        )

        # Advantage Stream
        self.advantage_stream = nn.Sequential(
            NoisyLinear(fc_input_dim, 512),
            nn.ReLU(),
            NoisyLinear(512, num_actions)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Sample taus
        taus = torch.rand(batch_size, self.n_quantiles, device=x.device)
        phi = self.quantile_embedding(taus)

        features = self.feature_extractor(x)  # shape: [B, 64, 6, 6]
        features = features.view(batch_size, -1).unsqueeze(1).expand(-1, self.n_quantiles, -1)
        x = phi * features  # Hadamard product

        # Flatten quantile dimension into batch
        x = x.view(batch_size * self.n_quantiles, -1)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values.view(batch_size, self.n_quantiles, -1)
