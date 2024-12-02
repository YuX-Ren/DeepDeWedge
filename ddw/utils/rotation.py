import math

import torch
from scipy import ndimage, spatial
import torch.nn.functional as F


def rotate_vol_around_axis(vol, rot_angle, rot_axis, output_shape=None, order=3):
    """
    Rotates the 3D tensor 'vol' by 'rot_angle' degrees around 'rot_axis'. The rotated tensor, which is typically larger than the original one, is center-cropped such that it has dimensions 'output_shape'. If 'output_shape' is None, the rotated tensor is cropped to the dimensions of 'vol'.
    """
    vol_shape = torch.tensor(vol.shape[-3:])
    if output_shape is None:
        output_shape = vol_shape
    # need later for cropping
    crop_offset = [math.floor((vs - cs) / 2) for vs, cs in zip(vol_shape, output_shape)]
    if rot_angle != 0:
        if not torch.is_tensor(rot_angle):
            rot_angle = torch.tensor(rot_angle)
        rot_angle = torch.deg2rad(rot_angle)
        # convert rotation axis and angle to a 3x3 rotation matrix
        rot_axis = rot_axis.float()
        rot = spatial.transform.Rotation.from_rotvec(
            rot_angle * (rot_axis / rot_axis.norm())
        )
        rot_mat = rot.as_matrix()
        # determine offset to rotate around center of volume
        # see https://stackoverflow.com/questions/20161175/how-can-i-use-scipy-ndimage-interpolation-affine-transform-to-rotate-an-image-ab
        # -1 because indexing starts at 0
        c_in = 0.5 * (vol_shape - torch.ones(3)).float().numpy()
        offset = c_in - rot_mat @ c_in
        # apply the rotation using affine_transform
        vol = torch.tensor(
            ndimage.affine_transform(vol, matrix=rot_mat, offset=offset, order=order),
            device=vol.device,
            dtype=vol.dtype,
        )
    vol = vol[
        crop_offset[0] : crop_offset[0] + output_shape[0],
        crop_offset[1] : crop_offset[1] + output_shape[1],
        crop_offset[2] : crop_offset[2] + output_shape[2],
    ]
    return vol


def rotate_vol_around_axis_GPU(vol, rot_angle, rot_axis, output_shape=None, order=3):
    """
    Rotates the 3D tensor 'vol' by 'rot_angle' degrees around 'rot_axis'. The rotated tensor, which is typically larger than the original one, is center-cropped such that it has dimensions 'output_shape'. If 'output_shape' is None, the rotated tensor is cropped to the dimensions of 'vol'.
    """
    vol_shape = torch.tensor(vol.shape[-3:], device=vol.device)
    if output_shape is None:
        output_shape = vol_shape
    # Compute the crop offset for center cropping
    crop_offset = [math.floor((vs - cs) / 2) for vs, cs in zip(vol_shape, output_shape)]
    if rot_angle != 0:
        if not torch.is_tensor(rot_angle):
            rot_angle = torch.tensor(rot_angle, device=vol.device, dtype=torch.float32)
        rot_angle = torch.deg2rad(rot_angle)
        # Normalize rotation axis and compute rotation vector
        rot_axis = rot_axis.to(device=vol.device, dtype=torch.float32)
        rot_axis = rot_axis / rot_axis.norm()
        rot_vec = rot_angle * rot_axis
        # Compute rotation matrix using scipy and convert to torch tensor
        rot = spatial.transform.Rotation.from_rotvec(rot_vec.cpu().numpy())
        rot_mat = torch.tensor(rot.as_matrix(), device=vol.device, dtype=torch.float32)
        # Prepare affine matrix for grid_sample (shape: [1, 3, 4])
        affine_mat = torch.zeros(1, 3, 4, device=vol.device, dtype=torch.float32)
        affine_mat[0, :, :3] = rot_mat
        # Add batch and channel dimensions to vol (shape: [1, 1, D, H, W])
        vol = vol.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
        # Create affine grid and perform sampling
        grid = F.affine_grid(affine_mat, size=vol.shape, align_corners=False)
        vol = F.grid_sample(vol, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        # Remove batch and channel dimensions
        vol = vol.squeeze(0).squeeze(0).to(dtype=vol.dtype)
    # Center crop the volume to the desired output shape
    vol = vol[
        crop_offset[0]: crop_offset[0] + output_shape[0],
        crop_offset[1]: crop_offset[1] + output_shape[1],
        crop_offset[2]: crop_offset[2] + output_shape[2],
    ]
    return vol

