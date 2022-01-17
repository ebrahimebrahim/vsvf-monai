# Copyright 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import numpy as np

class VsvfDualizeMomentum(nn.Module):
  """
  A layer used to dualize a momentum vector field and obtain a velocity vector field, as seen in the
  momentum-parameterized LDDMM shooting approach to fluid image registration.
  """
  def __init__(self,):
    super().__init__()
    pass

  def forward(self, x):
    return x

class VsvfIntegrateVelocity(nn.Module):
  """
  A layer used to integrate a given velocity vector field to obtain a displacement field, as seen in mermaid.
  """
  def __init__(self,):
    super().__init__()

  def forward(self, x):
    return x

class Vsvf(nn.Module):
  """
  A block that can be used for the momentum-based approach to fluid image registration via stationary velocity fields,
  based on "Shen et al., Networks for Joint Affine and Non-parametric Image Registration <https://arxiv.org/abs/1903.08811>."
  Given a momentum vector field, the chosen regularizer is applied to obtain a velocity field, and the velocity field
  is integrated to obtain a displacement field.
  """

  def __init__(self,):
    super().__init__()
    self.dualizer = VsvfDualizeMomentum()
    self.integrator = VsvfIntegrateVelocity()

  def forward(self, x):
    x = self.dualizer(x) # get velocity field
    x = self.integrator(x) # integrate to get displacement field
    return x