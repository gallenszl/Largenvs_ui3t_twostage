import omegaconf
import importlib 
from torch.utils.data import DataLoader
from einops import rearrange
import imageio
import numpy as np

import random 
random.seed(1)

# config = omegaconf.OmegaConf.load('configs/LVSM_scene_decoder_only.yaml')
config = omegaconf.OmegaConf.load('configs/VGGT4LVSM_scene_decoder_only.yaml')
dataset_name = config.training.get("dataset_name")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=True,
)

batch = next(iter(dataloader))

images = batch['image']
images = rearrange(images, '1 n c h w -> h (n w) c')
images = (images.numpy() * 255).astype(np.uint8)
imageio.imsave('debug_image2.png', images)