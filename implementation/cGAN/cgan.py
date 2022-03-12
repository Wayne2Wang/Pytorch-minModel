import torch
import torch.nn as nn

# Specific architecture design is not the most important part

# in the original paper Maxout was used
class Discriminator(nn.Module):
    def __init__(self, img_dim, label_dim):
        super().__init__()
        self.fc_x = nn.Linear(img_dim, 800)
        self.fc_y = nn.Linear(label_dim, 200)
        self.fc1 = nn.Linear(1000, 500)
        self.fc2 = nn.Linear(500, 1)
        self.relu = nn.ReLU()#nn.LeakyReLU(0.01)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, y):
        x = self.dropout(x)
        x = self.relu(self.fc_x(x))
        y = self.relu(self.fc_y(y))
        combined = torch.cat([x, y], 1)
        combined = self.dropout(combined)
        combined = self.relu(self.fc1(combined))
        combined = self.dropout(combined)
        combined = self.sigmoid(self.fc2(combined))
        return combined

# Tanh is very important for the model to work
class Generator(nn.Module):
    def __init__(self, noise_dim, img_dim, label_dim):
        super().__init__()
        self.fc_z = nn.Linear(noise_dim, 600)
        self.fc_y = nn.Linear(label_dim, 200)
        self.fc1 = nn.Linear(800, 800)
        self.fc2 = nn.Linear(800, img_dim)
        self.relu = nn.ReLU()#nn.LeakyReLU(0.01)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, z, y):
        z = self.relu(self.fc_z(z))
        y = self.relu(self.fc_y(y))
        combined = torch.cat([z, y], 1)
        combined = self.relu(self.fc1(combined))
        combined = self.sigmoid(self.fc2(combined))
        return combined