import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase
import sys

# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth


    def forward(
        self,
        ray_bundle,
        device
    ):
        

        # TODO (1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        # Generates a set of distances between near and far
        z_vals = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray).view(1, -1, 1).to(device)
        N = ray_bundle.directions.shape[0]
        d = z_vals.shape[1]
        z_vals = z_vals.repeat(N,1,1)


        # TODO (1.4):  Uses these distances to sample points offset from ray origins (RayBundle.origins) along ray directions (RayBundle.directions). 

        sample_points = z_vals * ray_bundle.directions.view(-1, 1, 3) + ray_bundle.origins.view(-1, 1, 3)
        # print(f"z_vals shape: {z_vals.shape}, ray_bundle.directions shape: {ray_bundle.directions.shape}, ray_bundle.origins shape: {ray_bundle.origins.shape}")

     

        # Stores the distances and sample points in RayBundle.sample_points and RayBundle.sample_lengths
        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}