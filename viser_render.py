import importlib
from setup import init_config
import viser 
from viser import transforms as vtf
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np 
from einops import rearrange, repeat


bg_img = np.linspace(250, 200, 128).astype(int)
bg_img = repeat(bg_img, 'n -> n 128 3')

class Viewer:
    def __init__(self, config, dataset, model):
        self.config = config
        self.dataset = dataset
        self.model = model

        self.server = viser.ViserServer()
        self.server.scene.set_up_direction('-y')
        self.server.scene.set_background_image(bg_img)
        self._init_ui()
        self.draw_frame()

    def _init_ui(self):
        self.current_batch_id = 0
        self.slider = self.server.gui.add_slider(
            label='BatchID', min=0, max=len(self.dataset)-1, 
            step=1, initial_value=0)
        @self.slider.on_update
        def _(_):
            self.current_batch_id = int(self.slider.value)
            self.draw_frame()

        self.next_botton = self.server.gui.add_button(label='  Next ->')
        @self.next_botton.on_click
        def _(_):
            self.current_batch_id = (self.current_batch_id + 1) % len(self.dataset)
            self.slider.value = self.current_batch_id
            self.draw_frame()

        self.prev_botton = self.server.gui.add_button(label='<- Prev  ')
        @self.prev_botton.on_click
        def _(_):
            self.current_batch_id = (self.current_batch_id - 1) % len(self.dataset)
            self.slider.value = self.current_batch_id
            self.draw_frame()

        self.coord_render = self.server.scene.add_transform_controls(
            'cam_render', scale=0.3)
        @self.coord_render.on_update
        def _(_):
            self.render_frame()
        
        self.frame_render = self.server.scene.add_camera_frustum(
            'cam_render/frame', fov=1, aspect=1, scale=0.1, 
            position=np.array([0, -0.2, 0]), color=(200, 20, 20))
        
        self.server.scene.add_spline_catmull_rom(
            'cam_render/spline', points=np.array([[0,0,0], [0, -0.2, 0]]),
            color=(200, 20, 20))

    def draw_frame(self):
        batch = self.dataset[self.current_batch_id]
        for idx in range(2):
            img = batch["image"][idx].cpu().numpy()
            img = (rearrange(img, 'c h w -> h w c') * 255).astype(np.uint8)
            
            c2w = batch['c2w'][idx].numpy()
            wxyz_xyz = vtf.SE3.from_matrix(c2w).wxyz_xyz

            self.server.scene.add_camera_frustum(
                f'cam_{idx}', fov=1, aspect=1, image=img, scale=0.1,
                wxyz=wxyz_xyz[:4], position=wxyz_xyz[4:])
        
        self.render_frame()
    
    def render_frame(self):
        target_c2w = vtf.SE3(np.concatenate([self.coord_render.wxyz, self.coord_render.position], 0)).as_matrix()
        target_c2w = rearrange(torch.from_numpy(target_c2w), 'i j -> 1 1 i j').float().cuda()

        with torch.no_grad(), torch.autocast(
            enabled=config.training.use_amp,
            device_type="cuda",
            dtype=amp_dtype_mapping[config.training.amp_dtype],
        ):
            batch = self.dataset[self.current_batch_id]
            batch = {k: v.cuda()[:2].unsqueeze(0) if type(v) == torch.Tensor else v for k, v in batch.items()}
            target_img = model.forward_single_target_view(batch, target_c2w)
            target_img = rearrange(target_img.float(), '1 1 c h w -> h w c').detach().cpu().numpy()
            target_img = (target_img * 255).astype(np.uint8)

        self.frame_render.image = target_img


config = init_config()
dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)

module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config).cuda()
model.load_ckpt(config.training.checkpoint_dir)
model.eval()

torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype_mapping = {
    "fp16": torch.float16, 
    "bf16": torch.bfloat16, 
    "fp32": torch.float32, 
    'tf32': torch.float32
}

viewer = Viewer(config, dataset, model)
input('Press Enter to exit')