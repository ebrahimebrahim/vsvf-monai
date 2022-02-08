import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from mermaid.module_parameters import ParameterDict
from mermaid.registration_networks import SVFVectorMomentumMapNet

# file paths
SRC_IMG_PATH = os.path.join("data", "OAI", "img",
    "9352883_20051123_SAG_3D_DESS_LEFT_016610798103_image.nii.gz")
SRC_LABEL_PATH = os.path.join("data", "OAI", "label",
    "9352883_20051123_SAG_3D_DESS_LEFT_016610798103_label_all.nii.gz")
TGT_IMG_PATH = os.path.join("data", "OAI", "img",
    "9403165_20060316_SAG_3D_DESS_LEFT_016610900302_image.nii.gz")
TGT_LABEL_PATH = os.path.join("data", "OAI", "label",
    "9403165_20060316_SAG_3D_DESS_LEFT_016610900302_label_all.nii.gz")
INIT_PHI_PATH = os.path.join("data", "OAI", "init_phi",
    "9352883_9403165_init_phi.pt")
MOMENTUM_PATH = os.path.join("data", "OAI", "momentum",
    "9352883_9403165_m.pt")
VELOCITY_PATH = os.path.join("data", "OAI", "velocity",
    "9352883_9403165_v.pt")
PHI_PATH = os.path.join("data", "OAI", "phi", "9352883_9403165_phi.pt")

# load files
src_img = sitk.ReadImage(SRC_IMG_PATH)
src_img = sitk.GetArrayFromImage(src_img)
src_img = torch.Tensor(src_img)[None, None, :].cuda()   # D x H x W -> N x C x D x H x W
src_label = sitk.ReadImage(SRC_LABEL_PATH)
src_label = sitk.GetArrayFromImage(src_label)
src_label = torch.Tensor(src_label)[None, None, :].cuda()   # D x H x W -> N x C x D x H x W
tgt_img = sitk.ReadImage(TGT_IMG_PATH)
tgt_img = sitk.GetArrayFromImage(tgt_img)
tgt_img = torch.Tensor(tgt_img)[None, None, :].cuda()   # D x H x W -> N x C x D x H x W
tgt_label = sitk.ReadImage(TGT_LABEL_PATH)
tgt_label = sitk.GetArrayFromImage(tgt_label)
tgt_label = torch.Tensor(tgt_label)[None, None, :].cuda()   # D x H x W -> N x C x D x H x W
init_phi = torch.load(INIT_PHI_PATH)
ref_momentum_map = torch.load(MOMENTUM_PATH)
ref_velocity_map = torch.load(VELOCITY_PATH)
ref_phi = torch.load(PHI_PATH)

# set up advection solver
mntm_size = np.array([1, 1, 40, 96, 96])
mntm_spacing = 1. / (mntm_size[2::] - 1)
params = ParameterDict()
params['env'] = ({})
params['env']['get_momentum_from_external_network'] = True
params['forward_model'] = ({})
params['forward_model']['adjoin_on'] = True
params['forward_model']['atol'] = 1e-5
params['forward_model']['number_of_time_steps'] = 10
params['forward_model']['rtol'] = 1e-5
params['forward_model']['smoother'] = ({})
params['forward_model']['smoother']['multi_gaussian_stds'] = np.array([0.05, 0.1, 0.15, 0.2, 0.25])
params['forward_model']['smoother']['multi_gaussian_weights'] = np.array([0.06666666666666667, 0.13333333333333333, 0.19999999999999998, 0.26666666666666666, 0.3333333333333333])
params['forward_model']['smoother']['type'] = 'multiGaussian'
params['forward_model']['smoother_for_forward'] = ({})
params['forward_model']['smoother_for_forward']['type'] = 'multiGaussian'
params['forward_model']['solver'] = 'rk4'
params['forward_model']['tFrom'], params['forward_model']['tTo'] = 0.0, 1.0
params['load_velocity_from_forward_model'] = True
params['shooting_vector_momentum'] = ({})
params['shooting_vector_momentum']['use_velocity_mask_on_boundary'] = False
adv_solver = SVFVectorMomentumMapNet(mntm_size, mntm_spacing, params)

# solve advection eq.
adv_solver.m = ref_momentum_map
phi = adv_solver(init_phi.permute(0, 4, 1, 2, 3), None)
upsampled_phi = nn.functional.interpolate(phi, size=src_img.shape[2:],
    mode='trilinear', align_corners=True)
# N x C x D x H x W -> N x D x H x W x C
upsampled_phi = upsampled_phi.permute(0, 2, 3, 4, 1)
assert torch.abs(upsampled_phi - ref_phi).mean() < 1e-8

# check velocity
advection_vars = adv_solver.get_variables_to_transfer_to_loss_function()
velocity = advection_vars['initial_velocity']
assert torch.abs(velocity - ref_velocity_map).mean() < 1e-8

# warp label
warped_src_label = nn.functional.grid_sample(src_label.float(), upsampled_phi,
    mode='nearest', padding_mode="border", align_corners=True)

# compute dice
"""
Should get something like
=====================
class 0 Dice is 0.997
class 1 Dice is 0.655
class 2 Dice is 0.630
=====================
"""
from monai.metrics import compute_meandice
print("=====================")
for i in range(3):
    dice = compute_meandice(warped_src_label == i,
                            tgt_label == i,
                            include_background=False)
    print("class {} Dice is {:.3f}".format(i, dice.item()))
print("=====================")
