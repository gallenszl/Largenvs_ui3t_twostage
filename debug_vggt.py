import torch
from model.vggt.models.vggt import VGGT4LVSM
import omegaconf
import importlib 
from torch.utils.data import DataLoader

config = omegaconf.OmegaConf.load('configs/VGGT4LVSM_scene_decoder_only.yaml')
config.model.image_tokenizer.image_size = 256
config.model.image_tokenizer.patch_size = 8
config.model.target_pose_tokenizer.image_size = 256
config.model.target_pose_tokenizer.patch_size = 8

model = VGGT4LVSM(config=config, is_debugging=False)
model.eval()
model.cuda()


dataset_name = config.training.get("dataset_name", "data.dataset_scene_video.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=True,
)

batch = next(iter(dataloader))
batch = {k: v.cuda() if type(v) == torch.Tensor else v for k, v in batch.items()}


with torch.no_grad():
    ret_dict = model(batch)

def print_shape(x):
    if isinstance(x, torch.Tensor):
        return f"{x.shape}"
    elif isinstance(x, dict):
        return [f'{k}: {print_shape(v)}' for k,v in x.items()]
# print([f'{k},{v.shape}' for k,v in ret_dict.items()])

print(print_shape(ret_dict))