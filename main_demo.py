# In this script, we will demonstrate the use of the vSVF block by
# - getting some data (e.g. OASIS, OAI, synethetic data, whatever) and constructing dataloaders
# - training an affine registration model to pre-align images by affine transform
# - constructing and training a fluid registration model that uses our vSVF unit

# The vSVF unit will be a MONAI block that we create in `vsvf.py`.
# We will try to limit needed dependencies outside of MONAI; we include mermaid as a dependency in this repository for validation only.

# Once it's working, the vSVF block can be contributed to MONAI and this notebook can be converted into
# a tutorial in the style of the other [MONAI tutorials](https://github.com/Project-MONAI/tutorials).


import monai
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence

# the part we intend to contribute
from vsvf import Vsvf, VsvfDualizeMomentum, VsvfIntegrateVelocity

# for validation only
import mermaid

def synthesize_image(shape:Sequence[int], min_size:int, max_size:int):
  """
  Create a random test image.
  It will be a binary image of a shape. The shape can be an ellipsoid or a cuboid.
  The extents along each dimension are selected from the integers in [min_size, max_size).
  """
  if min_size < 1:
    raise ValueError("min_size should be positive")
  if max_size <= min_size:
    raise ValueError("min_size should be < max_size")

  image = np.zeros(shape)
  start_indices = [np.random.randint(0,d-min_size) for d in shape]
  stop_indices = [
    np.random.randint(start_index+min_size, min(start_index+max_size,d))
    for start_index,d in zip(start_indices,shape)
  ]

  if np.random.randint(2)==1: # make an ellipse/ellipsoid/etc
    open_mesh_grid = np.ogrid[[slice(0,d) for d in shape]]
    centers = [(start_index + stop_index)/2. for start_index, stop_index in zip(start_indices, stop_indices)]
    half_extents = [(stop_index-start_index)/2. for start_index, stop_index in zip(start_indices, stop_indices)]
    normalized_coords = [(open_mesh_grid[i]-centers[i])/half_extents[i] for i in range(len(shape))]
    ellipsoid_mask = sum(x**2 for x in normalized_coords) <= 1.0
    image[ellipsoid_mask]=1

  else: # make a rectangle/cuboid/etc
    image[tuple(slice(start_index, stop_index) for start_index, stop_index in zip(start_indices, stop_indices))]=1

  return image



if __name__ == "__main__":
  pass # main demo work will go here