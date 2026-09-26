import importlib
from setup import init_config
import viser 
from viser import transforms as vtf
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np 
from einops import rearrange, repeat
from recover_gravity import find_best_x_rotation_zero_yaw

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
        self.update_render_position()
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

        self.azimuth_slider = self.server.gui.add_slider(
            label='Azimuth', min=-180, max=180, step=20, initial_value=0)
        @self.azimuth_slider.on_update
        def _(_):
            self.update_render_position()
            self.render_frame()
        
        self.elevation_slider = self.server.gui.add_slider(
            label='Elevation', min=-90, max=90, step=10, initial_value=0)
        @self.elevation_slider.on_update
        def _(_):
            self.update_render_position()
            self.render_frame()

        self.radius_slider = self.server.gui.add_slider(
            label='Radius', min=0.75, max=1.5, step=0.05, initial_value=1)
        @self.radius_slider.on_update
        def _(_):
            self.update_render_position()
            self.render_frame()
                
    @staticmethod
    def get_camera_pose(azimuth, elevation, radius=1):
        x = np.sin(azimuth) * np.cos(elevation)
        y = np.sin(elevation)
        z = np.cos(azimuth) * np.cos(elevation)
        eye = np.array([x, y, z]) * radius

        target = np.array([0,0,0])
        up = np.array([0,-1,0])
        forward = (target - eye)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        true_up = np.cross(forward, right)
        true_up /= np.linalg.norm(true_up)
        rot_matrix = np.stack([right, true_up, forward], axis=1)
        wxyz = vtf.SO3.from_matrix(rot_matrix).wxyz

        return eye, wxyz

    def draw_frame(self):
        batch = self.dataset[self.current_batch_id]

        ### recover gravity
        c2w = batch['c2w'].numpy()
        theta, *_ = find_best_x_rotation_zero_yaw(c2w)
        transforms_so3 = vtf.SO3.from_x_radians(theta)
        transforms_se3 = vtf.SE3.from_rotation(transforms_so3).as_matrix()
        self.gravity_rectification = vtf.SE3.from_rotation(vtf.SO3.from_x_radians(-theta)).as_matrix()
        c2w = transforms_se3[None, ...] @ c2w
        batch['c2w'] = torch.from_numpy(c2w)

        num_input_imgs = 4
        for idx in range(num_input_imgs):
            img = batch["image"][idx].cpu().numpy()
            img = (rearrange(img, 'c h w -> h w c') * 255).astype(np.uint8)
            
            c2w = batch['c2w'][idx].numpy()
            wxyz_xyz = vtf.SE3.from_matrix(c2w).wxyz_xyz

            self.server.scene.add_camera_frustum(
                f'cam_{idx}', fov=1, aspect=1, image=img, scale=0.1,
                wxyz=wxyz_xyz[:4], position=wxyz_xyz[4:])
        
        self.render_frame()
    
    def update_render_position(self):
        azimuth = - self.azimuth_slider.value / 180 * np.pi - np.pi
        elevation = - self.elevation_slider.value / 180 * np.pi
        radius = self.radius_slider.value
        position, wxyz = self.get_camera_pose(azimuth, elevation, radius)
        if hasattr(self, 'cam_render'):
            self.cam_render.position = position
            self.cam_render.wxyz = wxyz
        else:
            self.cam_render = self.server.scene.add_camera_frustum(
                'cam_render', fov=1, aspect=1, scale=0.1, 
                position=position, wxyz=wxyz, color=(200, 20, 20))

    def render_frame(self):
        target_c2w = vtf.SE3(np.concatenate([self.cam_render.wxyz, self.cam_render.position], 0)).as_matrix()
        target_c2w = self.gravity_rectification @ target_c2w
        target_c2w = rearrange(torch.from_numpy(target_c2w), 'i j -> 1 1 i j').float().cuda()

        with torch.no_grad(), torch.autocast(
            enabled=config.training.use_amp,
            device_type="cuda",
            dtype=amp_dtype_mapping[config.training.amp_dtype],
        ):
            batch = self.dataset[self.current_batch_id]
            batch = {k: v.cuda()[:4].unsqueeze(0) if type(v) == torch.Tensor else v for k, v in batch.items()}
            render_pack = model.forward_single_target_view_unposed(batch, target_c2w)
            target_img = render_pack.render
            target_img = rearrange(target_img.float(), '1 1 c h w -> h w c').detach().cpu().numpy()
            target_img = (target_img * 255).astype(np.uint8)

            pts = render_pack.points
            pts_conf = render_pack.points_conf

            pts = rearrange(pts, '1 1 c h w -> h w c').detach().cpu().numpy()
            pts_conf = rearrange(pts_conf, '1 1 1 h w -> h w').detach().cpu().numpy()

            # print(pts_conf.min(), pts_conf.max(), pts_conf.mean())
            mask = np.logical_and((pts_conf > 10), (np.abs(pts).sum(-1) > 0))
            pts = self.gravity_rectification[:3,:3].T @ rearrange(pts[mask], 'n c -> c n')
            pts = rearrange(pts, 'c n -> n c')
            pts_color = target_img[mask]

            self.server.scene.add_point_cloud('pcd', points=pts, colors=pts_color,
                point_size=0.005, point_shape='circle')


        self.cam_render.image = target_img

config = init_config()
# dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
dataset_name = config.training.get("val_dataset_name")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)

model = None
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
exit()