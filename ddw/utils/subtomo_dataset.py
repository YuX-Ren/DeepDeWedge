import os

import torch
from scipy import spatial
from torch.utils.data import Dataset

from .fourier import apply_fourier_mask_to_tomo
from .missing_wedge import (get_missing_wedge_mask,
                            get_rotated_missing_wedge_mask)
from .rotation import rotate_vol_around_axis_GPU
from .noise_generator import noise_fbp
import math
BASE_SEED = 888
rotation_list_all = [(((0,1),1),((1,2),0)), (((0,1),1),((1,2),1)), (((0,2),1),((1,2),0)), (((0,2),1),((1,2),1)),
                (((0,1),1),((1,2),2)), (((0,1),1),((1,2),3)), (((0,2),1),((1,2),2)), (((0,2),1),((1,2),3)),
                (((0,1),3),((1,2),0)), (((0,1),3),((1,2),1)), (((0,2),3),((1,2),0)), (((0,2),1),((1,2),1)),
                (((0,1),3),((1,2),2)), (((0,1),3),((1,2),3)), (((0,2),3),((1,2),2)), (((0,2),1),((1,2),3)),
                (((1,2),1),((0,2),0)), (((1,2),1),((0,2),2)), (((1,2),3),((0,2),0)), (((1,2),3),((0,2),2)),
                (((1,2),1),((0,1),0)), (((1,2),1),((0,1),2)), (((1,2),3),((0,1),0)), (((1,2),3),((0,1),2))]


class SubtomoDataset(Dataset):
    """
    A torch dataset which produces the input-target sub-tomogram pairs used for model fitting. The directory 'subtomo_dir' must have the same structure as the output of the 'ddw prepare-data' command.
    """

    def __init__(
        self,
        subtomo_dir,
        mw_angle,
        crop_subtomos_to_size,
        rotate_subtomos=True,
        deterministic_rotations=False,
    ):
        super().__init__()
        self.subtomo_dir = subtomo_dir
        self.crop_subtomos_to_size = crop_subtomos_to_size
        self.mw_angle = mw_angle
        self.rotate_subtomos = rotate_subtomos
        self.deterministic_rotations = deterministic_rotations

    @property
    def rotate_subtomos(self):
        return self._rotate_subtomos

    @rotate_subtomos.setter
    def rotate_subtomos(self, rotate_subtomos):
        if not isinstance(rotate_subtomos, bool):
            raise ValueError("rotate_subtomos must be a boolean")
        self._rotate_subtomos = rotate_subtomos

    def _sample_rot_axis_and_angle(self, index):
        seed = BASE_SEED + index if self.deterministic_rotations else None
        rotvec = torch.from_numpy(
            spatial.transform.Rotation.random(random_state=seed).as_rotvec()
        )
        rot_axis = rotvec / rotvec.norm()
        rot_angle = torch.rad2deg(rotvec.norm())
        return rot_axis, rot_angle

    def __len__(self):
        return len(os.listdir(f"{self.subtomo_dir}/subtomo0"))

    def __getitem__(self, index):
        # load subtomos
        subtomo0_file = f"{self.subtomo_dir}/subtomo0/{index}.pt"
        subtomo0 = torch.load(subtomo0_file)
        subtomo1_file = f"{self.subtomo_dir}/subtomo1/{index}.pt"
        subtomo1 = torch.load(subtomo1_file)
        # rotate subtomos
        if self.rotate_subtomos == True:
            # rot_axis, rot_angle = self._sample_rot_axis_and_angle(index)
            rot_axis, rot_angle = 0, 0
            # subtomo0 = rotate_vol_around_axis_GPU(
            #     subtomo0,
            #     rot_angle=rot_angle,
            #     rot_axis=rot_axis,
            #     output_shape=3 * [self.crop_subtomos_to_size],
            # )
            crop_offset = [math.floor((vs - cs) / 2) for vs, cs in zip(subtomo0.shape, 3 * [self.crop_subtomos_to_size])]
            subtomo0 = subtomo0[crop_offset[0]:crop_offset[0]+self.crop_subtomos_to_size, crop_offset[1]:crop_offset[1]+self.crop_subtomos_to_size, crop_offset[2]:crop_offset[2]+self.crop_subtomos_to_size]
            # random rotation in 24 face directions
            r = rotation_list_all[torch.randint(0, len(rotation_list_all), (1,)).item()]
            subtomo0 = torch.rot90(
                subtomo0, k=r[0][1], dims=[d for d in r[0][0]]
            )
            subtomo0 = torch.rot90(
                subtomo0, k=r[1][1], dims=[d for d in r[1][0]]
            )
            # subtomo1 = rotate_vol_around_axis_GPU(
            #     subtomo1,
            #     rot_angle=rot_angle,
            #     rot_axis=rot_axis,
            #     output_shape=3 * [self.crop_subtomos_to_size],
            # )
            # add missing wedge
            # mw_mask = get_missing_wedge_mask(
            #     grid_size=3 * [self.crop_subtomos_to_size],
            #     mw_angle=self.mw_angle,
            #     device=subtomo0.device,
            # )
            mw_mask = 0
            rot_mw_mask = mw_mask
            # rot_mw_mask = get_rotated_missing_wedge_mask(
            #     grid_size=3 * [self.crop_subtomos_to_size],
            #     mw_angle=self.mw_angle,
            #     rot_axis=rot_axis,
            #     rot_angle=rot_angle,
            #     device=subtomo0.device,
            # )
        else:
            mw_mask = get_missing_wedge_mask(
                grid_size=subtomo0.shape,
                mw_angle=self.mw_angle,
                device=subtomo0.device,
            )
            rot_mw_mask = mw_mask
            rot_angle, rot_axis = 0, torch.tensor([1.0, 0.0, 0.0])

        # model_input = apply_fourier_mask_to_tomo(subtomo0, mw_mask)
        model_input = subtomo0
        item = {
            "model_input": model_input,
            "model_target": subtomo1,
            "mw_mask": mw_mask,
            "rot_mw_mask": rot_mw_mask,
            "subtomo0_file": subtomo0_file,
            "subtomo1_file": subtomo1_file,
            "rot_angle": rot_angle,
            "rot_axis": rot_axis,
            "mw_angle": self.mw_angle,
        }
        return item


# %%
