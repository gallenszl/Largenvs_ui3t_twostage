import importlib
from omegaconf import OmegaConf
import torchinfo

config = OmegaConf.load("configs/VGGT4LVSM_obj_decoder_only.yaml")

module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config)

torchinfo.summary(model)