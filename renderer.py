import torch
import sys

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg,
        device
    ):
        super().__init__()


        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False
        self._device = device

    def _compute_weights(
        self,
        deltas,
        rays_density: torch.Tensor,
        eps: float = 1e-10
    ):

        print(f" ******* inside compute weights *******")
        #================================================================================================
        # # TODO (1.5): Compute transmittance using the equation described in the README
        # # transmittance T(x, x_ti) = T(x, x_ti-1) * exp(-σ_ti-1 * Δt_i-1)

        # self.T_prev = torch.ones(1).to(self._device)
        # # print(f" rays_density {rays_density}")
        # # print(f" deltas {deltas}")
        # # print(f" T_prev {self.T_prev.shape}")
        # self.T_prev= self.T_prev.to(self._device)

        # print(f" ray {rays_density.shape}, delta {deltas.shape}")
        # self.T_curr = self.T_prev * torch.exp(-rays_density * deltas)

        # print(f" shape of T_curr {self.T_curr.shape}")

        # # TODO (1.5): Compute weight used for rendering from transmittance and density
        # # weights = T * (1 - exp(-σ * Δt))
        # transmittance =  (1 - torch.exp(-rays_density * deltas))
        # print(f" transmittance {transmittance.shape}")
        # weights = self.T_curr * (1 - torch.exp(-rays_density * deltas))

        # self.T_prev = self.T_curr
        # # print(f"weights {weights}")
        # return weights

        #================================================================================================

        #  Note that for the first segment T = 1
        num_rays = rays_density.shape[0]
        num_samples = rays_density.shape[1]

        T_prev = torch.ones(num_rays, 1).to(self._device)
  

        T = []
        for i in range(num_samples):
            T_curr = T_prev * torch.exp(-rays_density[:, i] * deltas[:, i])
            T.append(T_curr)
            T_prev = T_curr

        T = torch.stack(T, dim=1)
        transmittance = (1 - torch.exp(-rays_density * deltas))
        print(f"shape of T {T.shape}, transmittance {transmittance.shape}")
        weights = T * transmittance

        return weights

        
    
    def _aggregate(
        self,
        weights: torch.Tensor,
        rays_feature: torch.Tensor
    ):
        # TODO (1.5): Aggregate (weighted sum of) features using weights
        # L(x,w) = sum ( weights) * (Le(x_ti,w))  )

        # print(f" weights shape: {weights.shape}") #torch.Size([32768, 65, 1])
        # print(f" rays_feature shape: {rays_feature.shape}") #torch.Size([2129920, 3])

        chunk_size = weights.shape[0]
        num_samples = weights.shape[1]

        rays_feature_reshape = rays_feature.view(chunk_size, num_samples, -1)
        rays_feature_reshape = rays_feature_reshape.squeeze(2)

        print(f"rays_feature_reshape shape: {rays_feature_reshape.shape}, weights shape: {weights.shape}")

        feature = torch.sum(weights * rays_feature_reshape, dim=1 )

        return feature

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle
    ):
        B = ray_bundle.shape[0]

        print(f" ******* inside renderer foward *******")
        

        # Process the chunks of rays.
        chunk_outputs = []

        

        for chunk_start in range(0, B, self._chunk_size):
            # print(f" ******* inside chunk *******")
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle, device=self._device )
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            implicit_output = implicit_fn(cur_ray_bundle)
            density = implicit_output['density'] #volume density
            feature = implicit_output['feature'] #color feature

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            print(f" shape of delta {deltas.shape}, and density:  {density.shape}")

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 
            print(f" weights shape: {weights.shape}")

            # TODO (1.5): Render (color) features using weights
            
            # perform summation in VolumeRenderer._aggregate.
            # Use weights, and aggregation function to render color and depth (stored in RayBundle.sample_lengths)
            feature = self._aggregate(weights, feature)
            print(f" feature shape: {feature.shape}")

            # TODO (1.5): Render depth map
            # print(f" shape of depth_values: {depth_values.shape}") #torch.Size([32768, 65)]
            #resize weight shape ([32768, 65, 1]) into ([32768, 65])
            weights = weights.view(self._chunk_size, -1)

            #element-wise multiply weight and depth value
            depth = self._aggregate(weights, depth_values)
            
            # print(f" depth shape: {depth.shape}")

            # Return
            cur_out = {
                'feature': feature,
                'depth': depth,
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    'volume': VolumeRenderer
}
