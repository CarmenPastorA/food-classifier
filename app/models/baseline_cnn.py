import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=101):
        super(BaselineCNN, self).__init__()
        
        # Define a simple CNN architecture
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Fully connected layer
        self.fc1 = nn.Linear(128 * 16 * 16, 512)  # Adjust the input dimensions after pooling
        self.fc2 = nn.Linear(512, num_classes)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Pass through the convolution layers
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 128 * 16 * 16)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
