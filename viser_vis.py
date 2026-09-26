import importlib
from setup import init_config
import viser 
from viser import transforms as vtf
import torch
import numpy as np 
from einops import rearrange, repeat


config = init_config()
dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)

batch = dataset[0]

bg_img = np.linspace(250, 200, 128).astype(int)
bg_img = repeat(bg_img, 'n -> n 128 3')

class Viewer:
    def __init__(self, dataset):
        self.dataset = dataset

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

        self.prev_botton = self.server.gui.add_button(label='<- Prev  ')
        @self.prev_botton.on_click
        def _(_):
            self.current_batch_id = (self.current_batch_id - 1) % len(self.dataset)
            self.slider.value = self.current_batch_id

    def draw_frame(self):
        batch = self.dataset[self.current_batch_id]
        print(self.current_batch_id, batch['image'].shape[0])
        for idx in range(batch['image'].shape[0]):
            img = batch["image"][idx].cpu().numpy()
            img = (rearrange(img, 'c h w -> h w c') * 255).astype(np.uint8)

            depth = batch['depth_map'][idx].cpu().numpy()
            pts = batch['point_map'][idx].cpu().numpy()
            pts = rearrange(pts, 'c h w -> h w c')

            mask = depth > 0 # h w
            pts = pts[mask]

            # pts_color = img[mask,:3]
            pts_color = ((pts+1)*128).astype(np.uint8)

            c2w = batch['c2w'][idx].numpy()
            wxyz_xyz = vtf.SE3.from_matrix(c2w).wxyz_xyz

            xyz_norm = f'{np.linalg.norm(wxyz_xyz[4:]):.2f}'

            self.server.scene.add_camera_frustum(
                f'cam_{idx}', fov=1, aspect=1, image=img, scale=0.1,
                wxyz=wxyz_xyz[:4], position=wxyz_xyz[4:])
            
            self.server.scene.add_label(
                f'label_{idx}', text=f'{idx} {xyz_norm}',
                position=wxyz_xyz[4:], wxyz=wxyz_xyz[:4])

            self.server.scene.add_point_cloud(
                f'pts_{idx}', points=pts, colors=pts_color, point_size=0.003)

viewer = Viewer(dataset)
input('Press Enter to exit')