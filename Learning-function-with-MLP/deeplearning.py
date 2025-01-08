import copy
# Numpy library
import numpy as np
# Plotting library
import matplotlib.pyplot as plt
# PyTorch libraries
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class MLP(nn.Module):
    def __init__(self, input_dim=1, out_dim=1, width=10,depth=5, activation='tanh'):
       super(MLP, self).__init__()

       # Want to ensure there is at least one hidden layer
       assert depth > 1
       self.depth=depth

       # Selecting the activation for hidden layers
       if activation == 'tanh':
         self.activation = nn.Tanh()
       elif activation == 'sin':
         self.activation = torch.sin
       elif activation == 'relu':
         self.activation = nn.ReLU()

       # Creat list of layers and include the first hidden layer
       MLP_list = [nn.Linear(input_dim, width)]

       # Remaining hidden layers
       for _ in range(depth - 2):
           MLP_list.append(nn.Linear(width, width))

       # Output layer
       MLP_list.append(nn.Linear(width, out_dim))

       # Adding list of layers as modules
       self.model = nn.ModuleList(MLP_list)

       # Weights initialization
       def init_weights(layer):
         if isinstance(layer, nn.Linear):
             nn.init.uniform_(layer.weight, -1, 1)
             if layer.bias is not None:
              nn.init.uniform_(layer.bias, -1, 1)
       self.model.apply(init_weights)


    # Defining forward mode of MLP model
    def forward(self, x):
     for i, layer in enumerate(self.model):
        # Apply activation only to hidden layers
        if i < self.depth-1:
            x = self.activation(layer(x))
        else:
            x = layer(x)
     return x

     # Number of network parameters
    def num_network_params(self):
         num_params = (width + 1)*input_dim
         for i in range(self.depth):
             if i < self.depth-2:
                num_params = num_params + (width + 1)*width
         num_params = num_params + (width + 1)*output_dim
         return num_params

    # Copy the constructor
    def deep_copy(self):
        return copy.deepcopy(self)
    

# Create custom dataset class
class CustomDataset(Dataset):
    def __init__(self, samples):
       """
       Initialize the CustomDataset with paired samples.
       Args:
           samples (list of tuples): A list of (x, y) pairs
               representing the dataset samples.
       """
       self.samples = torch.Tensor(samples).to(torch.float32)
    def __len__(self):
       """
       Returns the length of the dataset, i.e., the number of
           samples.
       """
       return len(self.samples)
    def __getitem__(self, idx):
       """
       Returns the sample pairs corresponding to the given list
           of indices.
       Args:
           indices (list): A list of indices to retrieve samples
for.
       Returns:
           list: A list of (x, y) pairs corresponding to the
               specified indices.
      """
       selected_samples = self.samples[idx]
       return selected_samples
