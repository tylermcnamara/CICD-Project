import torch.nn as nn
import torch.nn.functional as F
'''
Target Model - LeNet:
    The classic CNN modified to read 256x512 resolution images.
    Uses 2 convolution layers using ReLU followed by max pooling layers, 
    and 3 FCNNs to get its prediction. 
'''
class LeNet(nn.Module):
    # Constructor method
    def __init__(self):
        super(LeNet, self).__init__()
        # Initial convolution layer, gets raw input finds basic features
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        # Second convolution layer, gets input from pooling of the feature map of the first layer
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # Max pooling layer, can be used for both convolutional layers
        self.pooling = nn.MaxPool2d(2, 2)

        # First FCNN, converts the flattened feature maps into a vector
        self.fc1 = nn.Linear(16 * 62 * 126, 120)
        # Second FCNN, deepens features before classification
        self.fc2 = nn.Linear(120, 84)
        # Last FCNN, outputs class logits for prediction
        self.fc3 = nn.Linear(84, 2)

    # Foward method: computes one forward pass on the model
    def forward(self, x):
        # Apply max pooling to the result of the convolution layers
        x = self.pooling(F.relu(self.conv1(x)))
        x = self.pooling(F.relu(self.conv2(x)))
        # Flatten 3d feature maps into a 2d vector for linear networks
        x = x.view(-1, 16 * 62 * 126)
        # Compress data for a prediction
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # No activation function needed here, default is CEL which is perfect
        return self.fc3(x)