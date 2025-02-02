import os
import warnings

import hydra
import numpy as np
import torch
import tqdm
import imageio
import sys


from omegaconf import DictConfig
from PIL import Image
from pytorch3d.renderer import (
    PerspectiveCameras,
    look_at_view_transform
)
import matplotlib.pyplot as plt

from implicit import volume_dict
from sampler import sampler_dict
from renderer import renderer_dict
from ray_utils import (
    sample_images_at_xy,
    get_pixels_from_image,
    get_random_pixels_from_image,
    get_rays_from_pixels
)
from data_utils import (
    dataset_from_config,
    create_surround_cameras,
    vis_grid,
    vis_rays,
)
from dataset import (
    get_nerf_datasets,
    trivial_collate,
)

from render_functions import (
    render_points
)

from sampler import StratifiedRaysampler

'''
1. RENDER
python main.py --config-name=box
2.

3. NERF
python main.py --config-name=nerf_lego

'''

# Model class containing:
#   1) Implicit volume defining the scene
#   2) Sampling scheme which generates sample points along rays
#   3) Renderer which can render an implicit volume given a sampling scheme

class Model(torch.nn.Module):
    def __init__(
        self,
        cfg, 
        device
    ):
        super().__init__()

        # Get implicit function from config
        self.implicit_fn = volume_dict[cfg.implicit_function.type](
            cfg.implicit_function
        )

        # Point sampling (raymarching) scheme
        self.sampler = sampler_dict[cfg.sampler.type](
            cfg.sampler
        )

        # Initialize volume renderer
        self.renderer = renderer_dict[cfg.renderer.type](
            cfg.renderer,
            device
        )
    
    def forward(
        self,
        ray_bundle
    ):
        # Call renderer with
        #  a) Implicit volume
        #  b) Sampling routine

        return self.renderer(
            self.sampler,
            self.implicit_fn,
            ray_bundle
        )


def render_images(
    model,
    cameras,
    cfg,
    image_size,
    viz,
    save=False,
    file_prefix=''
):
    all_images = []
    device = list(model.parameters())[0].device
    print(f"device: {device}")

    for cam_idx, camera in enumerate(cameras):
        print(f'Rendering image {cam_idx}')

        torch.cuda.empty_cache()
        camera = camera.to(device)
        xy_grid = get_pixels_from_image(image_size, camera) # TODO (1.3): implement in ray_utils.py
        ray_bundle = get_rays_from_pixels(xy_grid, image_size, camera, device) # TODO (1.3): implement in ray_utils.py

        # print(f" ray bundle origin: {ray_bundle.origins.shape} direction: {ray_bundle.directions.shape}")
        # sys.exit()

        # TODO (1.3): Visualize xy grid using vis_grid
        if cam_idx == 0 and file_prefix == '':
            xy_grid_vis = vis_grid(xy_grid, image_size)
            if viz:
                plt.imshow(xy_grid_vis)
                plt.show()

        # TODO (1.3): Visualize rays using vis_rays
        if cam_idx == 0 and file_prefix == '':
            rays_vis = vis_rays(ray_bundle, image_size)
            if viz:
                plt.imshow(rays_vis)
                plt.show()
        
        # TODO (1.4): Implement point sampling along rays in the function StratifiedSampler() in sampler.py
        sampler_model = StratifiedRaysampler(cfg.sampler)
        ray_bundle_sampled = sampler_model(ray_bundle, device) #([65536, 65, 3])

        #reshape into ([65536 x 65, 3])
        pcloud_points = ray_bundle_sampled.sample_points.view(-1, 3)
        # print(f" pcloud_points: {pcloud_points.shape}")
        pcloud_points = pcloud_points.unsqueeze(0)
        # print(f" pcloud_points: {pcloud_points.shape}")


        print(f" cam_idx: {cam_idx} file_prefix: {file_prefix}")
        print(f"viz: {viz} save: {save}")
        

        # TODO (1.4): Visualize sample points as point cloud using render_points
        if cam_idx == 0 and file_prefix == '':
            save_filename = 'images/Q1_4.png'

            rendered_img = render_points(save_filename, pcloud_points, image_size, color=[0.7, 0.7, 1], device=device)

            if viz:
                plt.imshow(rendered_img)
                plt.show()
            
                
        # sys.exit()
        # TODO (1.5): Implement rendering in renderer.py
        out = model(ray_bundle)

        

        # Return rendered features (colors)
        image = np.array(
            out['feature'].view(
                image_size[1], image_size[0], 3
            ).detach().cpu()
        )
        all_images.append(image)

        # TODO (1.5): Visualize depth
        if cam_idx == 2 and file_prefix == '':
            depth_img = np.array(
                out['depth'].view(
                    image_size[1], image_size[0]
                ).detach().cpu()
            )
            plt.imsave("images/depth.png", depth_img)

        # Save
        if save:
            plt.imsave(
                f'{file_prefix}_{cam_idx}.png',
                image
            )
    
    return all_images


def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def render(
    cfg
):

    device = get_device()
    # Create model
    model = Model(cfg, device)
    model = model.cuda(); model.eval()

    print(model)
    # sys.exit()

    # Render spiral
    cameras = create_surround_cameras(3.0, n_poses=20)
    all_images = render_images(
        model, cameras, cfg, cfg.data.image_size, cfg.viz, save=True, file_prefix=''
    )
    imageio.mimsave('images/part_1.gif', [np.uint8(im * 255) for im in all_images])


def train(
    cfg
):
    device = get_device()
    # Create model
    model = Model(cfg, device)
    model = model.cuda(); model.train()

    # Create dataset 
    train_dataset = dataset_from_config(cfg.data)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )
    image_size = cfg.data.image_size

    # Create optimizer 
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr
    )

    # Render images before training
    cameras = [item['camera'] for item in train_dataset]
    render_images(
        model, cameras, cfg, image_size, cfg.viz,
        save=True, file_prefix='images/part_2_before_training'
    )

    # Train
    t_range = tqdm.tqdm(range(cfg.training.num_epochs))

    for epoch in t_range:
        for iteration, batch in enumerate(train_dataloader):
            image, camera, camera_idx = batch[0].values()
            image = image.cuda()
            camera = camera.cuda()

            # Sample rays
            xy_grid = get_random_pixels_from_image(cfg.training.batch_size, image_size, camera) # TODO (2.1): implement in ray_utils.py
            ray_bundle = get_rays_from_pixels(xy_grid, image_size, camera, device)
            rgb_gt = sample_images_at_xy(image, xy_grid)

            # Run model forward
            out = model(ray_bundle)

            # TODO (2.2): Calculate loss
            loss = torch.nn.functional.mse_loss(out['feature'], rgb_gt)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch % 10) == 0:
            t_range.set_description(f'Epoch: {epoch:04d}, Loss: {loss:.06f}')
            t_range.refresh()

    # Print center and side lengths
    print("Box center:", tuple(np.array(model.implicit_fn.sdf.center.data.detach().cpu()).tolist()[0]))
    print("Box side lengths:", tuple(np.array(model.implicit_fn.sdf.side_lengths.data.detach().cpu()).tolist()[0]))

    # Render images after training
    render_images(
        model, cameras, cfg, image_size, cfg.viz,
        save=True, file_prefix='images/part_2_after_training'
    )
    all_images = render_images(
        model, create_surround_cameras(3.0, n_poses=20), cfg, image_size, cfg.viz, file_prefix='part_2'
    )
    imageio.mimsave('images/part_2.gif', [np.uint8(im * 255) for im in all_images])


def create_model(cfg):
    # Create model
    device = get_device()
    model = Model(cfg, device)
    model.cuda(); model.train()

    # Load checkpoints
    optimizer_state_dict = None
    start_epoch = 0

    checkpoint_path = os.path.join(
        hydra.utils.get_original_cwd(),
        cfg.training.checkpoint_path
    )

    if len(cfg.training.checkpoint_path) > 0:
        # Make the root of the experiment directory.
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Resume training if requested.
        if cfg.training.resume and os.path.isfile(checkpoint_path):
            print(f"Resuming from checkpoint {checkpoint_path}.")
            loaded_data = torch.load(checkpoint_path)
            model.load_state_dict(loaded_data["model"])
            start_epoch = loaded_data["epoch"]

            print(f"   => resuming from epoch {start_epoch}.")
            optimizer_state_dict = loaded_data["optimizer"]

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
    )

    # Load the optimizer state dict in case we are resuming.
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        optimizer.last_epoch = start_epoch

    # The learning rate scheduling is implemented with LambdaLR PyTorch scheduler.
    def lr_lambda(epoch):
        return cfg.training.lr_scheduler_gamma ** (
            epoch / cfg.training.lr_scheduler_step_size
        )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=start_epoch - 1, verbose=False
    )

    return model, optimizer, lr_scheduler, start_epoch, checkpoint_path

def train_nerf(
    cfg
):

    device = get_device()
    # Create model
    model, optimizer, lr_scheduler, start_epoch, checkpoint_path = create_model(cfg)

    print(f"model {model}")
    # Load the training/validation data.
    train_dataset, val_dataset, _ = get_nerf_datasets(
        dataset_name=cfg.data.dataset_name,
        image_size=[cfg.data.image_size[1], cfg.data.image_size[0]],
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=trivial_collate,
    )

    # Run the main training loop.
    for epoch in range(start_epoch, cfg.training.num_epochs):
        t_range = tqdm.tqdm(enumerate(train_dataloader))

        for iteration, batch in t_range:
            image, camera, camera_idx = batch[0].values()
            image = image.cuda().unsqueeze(0)
            camera = camera.cuda()

            # Sample rays
            xy_grid = get_random_pixels_from_image(
                cfg.training.batch_size, cfg.data.image_size, camera
            )
            ray_bundle = get_rays_from_pixels(
                xy_grid, cfg.data.image_size, camera, device
            )
            rgb_gt = sample_images_at_xy(image, xy_grid)

            # Run model forward
            out = model(ray_bundle)

            # TODO (3.1): Calculate loss
            loss = torch.nn.functional.mse_loss(out['feature'], rgb_gt)

            # Take the training step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_range.set_description(f'Epoch: {epoch:04d}, Loss: {loss:.06f}')
            t_range.refresh()

        # Adjust the learning rate.
        lr_scheduler.step()

        # Checkpoint.
        if (
            epoch % cfg.training.checkpoint_interval == 0
            and len(cfg.training.checkpoint_path) > 0
            and epoch > 0
        ):
            print(f"Storing checkpoint {checkpoint_path}.")

            data_to_store = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }

            torch.save(data_to_store, checkpoint_path)

        # Render
        if (
            epoch % cfg.training.render_interval == 0
            and epoch > 0
        ):
            with torch.no_grad():
                test_images = render_images(
                    model, create_surround_cameras(4.0, n_poses=20, up=(0.0, 0.0, 1.0), focal_length=2.0),
                     cfg, cfg.data.image_size, cfg.viz, file_prefix='nerf'
                )

                # render_images(
        # model, create_surround_cameras(3.0, n_poses=20), cfg, image_size, cfg.viz, file_prefix='part_2'

                imageio.mimsave('images/part_3.gif', [np.uint8(im * 255) for im in test_images])


@hydra.main(config_path='./configs', config_name='sphere')
def main(cfg: DictConfig):
    print("--------- init ---------")
    os.chdir(hydra.utils.get_original_cwd())


    # print(f" arg:", cfg.implicit_function.sdf.type )



    if cfg.type == 'render':
        print(f"inside render")
        render(cfg)
    elif cfg.type == 'train':
        train(cfg)
    elif cfg.type == 'train_nerf':
        train_nerf(cfg)


if __name__ == "__main__":
    main()

