import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.DP_Diffusion.model.ema import ExponentialMovingAverage

class TargetNetwork(nn.Module):
    def __init__(self, input_dim):
        super(TargetNetwork, self).__init__()
        # input_dim: (num_in_channels, image_size, image_size)
        self.num_in_channels, self.image_size, _ = input_dim  
        self.conv = nn.Sequential(
            nn.Conv2d(self.num_in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        test_input = torch.zeros(1, self.num_in_channels, self.image_size, self.image_size)
        test_output = self.conv(test_input)
        self.output_size = test_output.view(1, -1).shape[1] 
        self.fc = nn.Sequential(
            nn.Linear(self.output_size, 32),
            nn.ReLU()
        )
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=0.1)
                nn.init.constant_(m.bias, 0.)
    
    def forward(self, x):
        if len(x.shape) == 3:  # (3, 32, 32)
            x = x.unsqueeze(0)  # (1, 3, 32, 32)
        x = self.conv(x)  # (1, 16, 16, 16)
        x = x.view(x.size(0), -1)  # (1, 4096)
        return self.fc(x)


class PredictionNetwork(nn.Module):
    def __init__(self, input_dim):
        super(PredictionNetwork, self).__init__()
        # input_dim: (num_in_channels, image_size, image_size)
        self.num_in_channels, self.image_size, _ = input_dim
        self.conv = nn.Sequential(
            nn.Conv2d(self.num_in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        test_input = torch.zeros(1, self.num_in_channels, self.image_size, self.image_size)
        test_output = self.conv(test_input)
        self.output_size = test_output.view(1, -1).shape[1] 
        self.fc = nn.Sequential(
            nn.Linear(self.output_size, 32),
            nn.ReLU()
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=0.1)
                nn.init.constant_(m.bias, 0.)
    
    def forward(self, x):
        if len(x.shape) == 3:  # (3, 32, 32)
            x = x.unsqueeze(0)  # (1, 3, 32, 32)
        x = self.conv(x)  # (1, 16, 16, 16)
        x = x.view(x.size(0), -1)  # (1, 4096)
        return self.fc(x)


class Rnd(nn.Module):
    def __init__(self, input_dim, device):
        super(Rnd, self).__init__()
        self.target_net = TargetNetwork(input_dim=input_dim).to(device)
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.prediction_net = PredictionNetwork(input_dim=input_dim).to(device)
        self.rnd_optimizer = torch.optim.Adam(self.prediction_net.parameters(), lr=0.001)

    def forward(self, x):
        with torch.no_grad():
            target_out = self.target_net(x)
        prediction_out = self.prediction_net(x)
        # Compute per-sample loss (reduction='none') and mean over output dimensions
        sample_losses = F.mse_loss(prediction_out, target_out, reduction='none').mean(dim=1)  # Shape: (batch_size,)
        # Compute batch mean loss for optimization
        rnd_loss = sample_losses.mean()  # Shape: scalar
        
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
        self.rnd_optimizer.step()
        
        return rnd_loss, sample_losses  # Return both for flexibility
