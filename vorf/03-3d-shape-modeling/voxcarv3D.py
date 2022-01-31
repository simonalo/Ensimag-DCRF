import math as m
import matplotlib.image as mpimg
import numpy as np
import skimage
from skimage import measure
import trimesh


# Camera Calibration for Al's image[1..12].pgm   
calib = np.array([
    [-230.924, 0, -33.6163, 300,  -78.8596, -178.763, -127.597, 300,
     -0.525731, 0, -0.85065, 2],
    [-178.763, -127.597, -78.8596, 300,  0, -221.578, 73.2053, 300,
     0, -0.85065, -0.525731, 2],
    [-73.2053, 0, -221.578, 300,  78.8596, -178.763, -127.597, 300,
     0.525731, 0, -0.85065, 2],
    [-178.763, 127.597, -78.8596, 300,  0, 33.6163, -230.924, 300,
     0, 0.85065, -0.525731, 2],
    [73.2053, 0, 221.578, 300,  -78.8596, -178.763, 127.597, 300,
     -0.525731, 0, 0.85065, 2],
    [230.924, 0, 33.6163, 300,  78.8596, -178.763, 127.597, 300,
     0.525731, 0, 0.85065, 2],
    [178.763, -127.597, 78.8596, 300,  0, -221.578, -73.2053, 300,
     0, -0.85065, 0.525731, 2],
    [178.763, 127.597, 78.8596, 300,  0, 33.6163, 230.924, 300,
     0, 0.85065, 0.525731, 2],
    [-127.597, -78.8596, 178.763, 300,  -33.6163, -230.924, 0, 300,
     -0.85065, -0.525731, 0, 2],
    [-127.597, 78.8596, 178.763, 300,  -221.578, -73.2053, 0, 300,
     -0.85065, 0.525731, 0, 2],
    [127.597, 78.8596, -178.763, 300,  221.578, -73.2053, 0, 300,
     0.85065, 0.525731, 0, 2],
    [127.597, -78.8596, -178.763, 300,  33.6163, -230.924, 0, 300,
     0.85065, -0.525731, 0, 2]
])


def generate_occupancy(X_user=None, Y_user=None, Z_user=None, resolution=100):
    # Build 3D grids
    # 3D Grids are of size: resolution x resolution x resolution/2
    step = 2 / resolution

    # Voxel coordinates
    if X_user is None:
        X, Y, Z = np.mgrid[-1:1:step, -1:1:step, -0.5:0.5:step]
    else:
        X, Y, Z = X_user.reshape((100, 100, 50)), Y_user.reshape((100, 100, 50)), Z_user.reshape((100, 100, 50))

    # Voxel occupancy
    occupancy = np.ndarray((resolution, resolution, resolution // 2), dtype=int)

    # Voxels are initially occupied then carved with silhouette information
    occupancy.fill(1) 

    for i in range(12):
        # read the input silhouettes
        myFile = "image{0}.pgm".format(i)
        print(myFile)
        img = mpimg.imread(myFile)
        if img.dtype == np.float32:  # if not integer
            img = (img * 255).astype(np.uint8)

        proj = calib[i].reshape((3, 4))
        mat_coord = [X.ravel(), Y.ravel(), Z.ravel(), np.ones_like(X.ravel())]
        u_v = np.dot(proj, mat_coord)
        u_v = u_v[:2] / u_v[2]
        u_v = u_v.astype('int32')
        u_v[0] = np.minimum(np.maximum(u_v[0], 0), img.shape[0] - 1)
        u_v[1] = np.minimum(np.maximum(u_v[1], 0), img.shape[1] - 1)

        img_pixels = img[u_v[1], u_v[0]]
        img_occupancy = img_pixels.reshape(occupancy.shape) > 0
        for i, line in enumerate(img_occupancy):
            for j, pixel in enumerate(line):
                occupancy[i, j] = occupancy[i, j] & img_occupancy[i, j]

    return X, Y, Z, occupancy


# ---------- MAIN ----------
if __name__ == "__main__":
    
    _, _, _, occupancy = generate_occupancy(100)

    # Voxel visualization

    # Use the marching cubes algorithm
    verts, faces, normals, values = measure.marching_cubes(occupancy, 0.25)

    # Export in a standard file format
    surf_mesh = trimesh.Trimesh(verts, faces, validate=True)
    surf_mesh.export('alvoxels.off')
 
