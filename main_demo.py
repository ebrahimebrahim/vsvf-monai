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

# the part we intend to contribute
from vsvf import Vsvf, VsvfDualizeMomentum, VsvfIntegrateVelocity

# for validation only
import mermaid


if __name__ == "__main__":
  pass # main demo work will go here