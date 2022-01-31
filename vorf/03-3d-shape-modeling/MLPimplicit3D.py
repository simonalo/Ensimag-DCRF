import math as m
from zlib import Z_RLE
import matplotlib.image as mpimg
import numpy as np
import skimage
from skimage import measure
import torch
from torch import nn
import trimesh

from voxcarv3D import generate_occupancy

# Camera Calibration for Al's image[1..12].pgm
calib = np.array([
    [-78.8596, -178.763, -127.597, 300, -230.924, 0, -33.6163, 300,
     -0.525731, 0, -0.85065, 2],
    [0, -221.578, 73.2053, 300, -178.763, -127.597, -78.8596, 300,
     0, -0.85065, -0.525731, 2],
    [ 78.8596, -178.763, -127.597, 300, -73.2053, 0, -221.578, 300,
     0.525731, 0, -0.85065, 2],
    [0, 33.6163, -230.924, 300, -178.763, 127.597, -78.8596, 300,
     0, 0.85065, -0.525731, 2],
    [-78.8596, -178.763, 127.597, 300, 73.2053, 0, 221.578, 300,
     -0.525731, 0, 0.85065, 2],
    [78.8596, -178.763, 127.597, 300, 230.924, 0, 33.6163, 300,
     0.525731, 0, 0.85065, 2],
    [0, -221.578, -73.2053, 300, 178.763, -127.597, 78.8596, 300,
     0, -0.85065, 0.525731, 2],
    [0, 33.6163, 230.924, 300, 178.763, 127.597, 78.8596, 300,
     0, 0.85065, 0.525731, 2],
    [-33.6163, -230.924, 0, 300, -127.597, -78.8596, 178.763, 300,
     -0.85065, -0.525731, 0, 2],
    [-221.578, -73.2053, 0, 300, -127.597, 78.8596, 178.763, 300,
     -0.85065, 0.525731, 0, 2],
    [221.578, -73.2053, 0, 300, 127.597, 78.8596, -178.763, 300,
     0.85065, 0.525731, 0, 2],
    [33.6163, -230.924, 0, 300, 127.597, -78.8596, -178.763, 300,
     0.85065, -0.525731, 0, 2]
])

# Training
MAX_EPOCH = 50
BATCH_SIZE = 100

# Build 3D grids
# 3D Grids are of size resolution x resolution x resolution/2
resolution = 100
step = 2 / resolution

# Voxel coordinates
X, Y, Z = np.mgrid[-1:1:step, -1:1:step, -0.5:0.5:step]

# Voxel occupancy
occupancy = np.ndarray((resolution, resolution, resolution // 2), dtype=int)

# Voxels are initially occupied then carved with silhouette information
occupancy.fill(1)


# MLP class
class MLP(nn.Module):
    """
    Multilayer Perceptron.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 60),
            nn.Tanh(),
            nn.Linear(60, 120),
            nn.ReLU(),
            nn.Linear(120, 60),
            nn.ReLU(),
            nn.Linear(60, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        """ Forward pass """
        return self.layers(x)


# GPU or not GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

    
# MLP Training
def nif_train(data_in, data_out, batch_size):
    # Initialize the MLP
    mlp = MLP()
    mlp = mlp.float()
    mlp.to(device)

    # Normalize cost between 0 and 1 in the grid
    n_one = (data_out == 1).sum()

    # loss for positives will be multiplied by this factor in the loss function
    p_weight = (data_out.size()[0] - n_one) / n_one
    print("Pos. Weight: ", p_weight)

    # Define the loss function and optimizer
    # loss_function = nn.CrossEntropyLoss()

    # sigmoid included in this loss function
    loss_function = nn.BCEWithLogitsLoss(pos_weight=p_weight)
    optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-2)

    # Run the training loop
    for epoch in range(0, MAX_EPOCH):

        print(f'Starting epoch {epoch + 1}/{MAX_EPOCH}')

        # Creating batch indices
        permutation = torch.randperm(data_in.size()[0])

        # Set current loss value
        current_loss = 0.0
        accuracy = 0

        # Iterate over batches
        for i in range(0, data_in.size()[0], batch_size):

            indices = permutation[i:i + batch_size]
            batch_x, batch_y = data_in[indices], data_out[indices]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Zero the gradient
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp(batch_x.float())

            # Compute loss
            loss = loss_function(outputs, batch_y.float())

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print current loss so far
            current_loss += loss.item()
            if (i/batch_size) % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                      ((i/batch_size) + 1, current_loss / (i/batch_size) + 1))

        outputs = torch.sigmoid(mlp(data_in.float()))
        acc = binary_acc(outputs, data_out)
        print("Binary accuracy: ", acc)

        # Training is complete.
    print('MLP trained.')
    return mlp


# IOU evaluation between binary grids
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    accuracy = correct_results_sum / y_test.shape[0]
    accuracy = torch.round(accuracy * 100)
    return accuracy


def main():
    # Generate X,Y,Z and occupancy
    X, Y, Z, _ = generate_occupancy(resolution=100)
    data_in_random = np.stack((X, Y, Z), axis=-1)
    resolution_cube = resolution * resolution * resolution
    data_in = np.reshape(data_in_random, (resolution_cube // 2, 3))

    # Pytorch format
    data_in = torch.from_numpy(data_in).to(device)

    # Random X, Y, Z
    x_size, y_size, z_size = X.shape
    total_size = x_size * y_size * z_size
    # Format data for PyTorch
    X_rand = np.random.default_rng().uniform(low=-1, high=1, size=total_size)
    Y_rand = np.random.default_rng().uniform(low=-1, high=1, size=total_size)
    Z_rand = np.random.default_rng().uniform(low=-0.5, high=0.5, size=total_size)
    _, _, _, occupancy = generate_occupancy(X_rand, Y_rand, Z_rand, 100)
    data_in_random = np.stack((X_rand, Y_rand, Z_rand), axis=-1)
    resolution_cube = resolution * resolution * resolution
    data_in_random = np.reshape(data_in_random, (resolution_cube // 2, 3))
    data_out = np.reshape(occupancy, (resolution_cube // 2, 1))

    # Pytorch format
    data_in_random = torch.from_numpy(data_in_random).to(device)
    data_out = torch.from_numpy(data_out).to(device)

    # Train mlp
    mlp = nif_train(data_in_random, data_out, BATCH_SIZE)  # data_out.size()[0])

    # Visualization on training data
    outputs = mlp(data_in.float())
    occ = outputs.detach().cpu().numpy()  # from torch format to numpy

    # Go back to 3D grid
    newocc = np.reshape(occ, (resolution, resolution, resolution // 2))
    newocc = np.around(newocc)

    # Marching cubes
    verts, faces, normals, values = measure.marching_cubes(newocc, 0.25)
    # Export in a standard file format
    surf_mesh = trimesh.Trimesh(verts, faces, validate=True)
    surf_mesh.export('alimplicit.off')


# --------- MAIN ---------
if __name__ == "__main__":
    main()
