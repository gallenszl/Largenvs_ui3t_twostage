import viser 
import os 
import numpy as np
import imageio
from os.path import join as opj
from einops import rearrange, repeat
from viser import transforms as vtf
import json

pred_folder = 'experiments/evaluation/test_obj_RnGUP_'
pred_list = sorted(os.listdir(pred_folder))
data_length = len(pred_list)


bg_img = np.linspace(250, 200, 128).astype(int)
bg_img = repeat(bg_img, 'n -> n 128 3')


color_by_id = [
    (38, 70, 83),
    (42, 157, 143),
    (244, 162, 97),
    (193, 56, 22)
]


class Viewer:
    def __init__(self):
        self.server = viser.ViserServer()
        self.server.scene.set_up_direction('-y')
        self.server.scene.set_background_image(bg_img)
        self._init_ui()
        self.draw_frame()
    
    def _init_ui(self):
        self.current_batch_id = 0
        self.slider = self.server.gui.add_slider(
            label='BatchID', min=0, max=data_length-1, 
            step=1, initial_value=0)
        @self.slider.on_update
        def _(_):
            self.current_batch_id = int(self.slider.value)
            self.draw_frame()

        self.next_botton = self.server.gui.add_button(label='  Next ->')
        @self.next_botton.on_click
        def _(_):
            self.current_batch_id = (self.current_batch_id + 1) % data_length
            self.slider.value = self.current_batch_id

        self.prev_botton = self.server.gui.add_button(label='<- Prev  ')
        @self.prev_botton.on_click
        def _(_):
            self.current_batch_id = (self.current_batch_id - 1) % data_length
            self.slider.value = self.current_batch_id

    def draw_frame(self):
        obj_name = pred_list[self.current_batch_id]
        obj_pred_folder = opj(pred_folder, obj_name)
        json_path = opj(obj_pred_folder, 'metrics_pose.json')
        infer_pose = json.load(open(json_path, 'r'))
        
        input_frames = imageio.imread(opj(obj_pred_folder, 'input.png'))
        input_frames = rearrange(input_frames, 'h (n w) c -> n h w c', n=4)

        gt_pose = np.array(infer_pose['pose']['gt_pose'])
        pred_pose = np.array(infer_pose['pose']['pred_pose'])

        for i in range(4):
            gt_se3 = vtf.SE3.from_matrix(gt_pose[i])
            pred_se3 = vtf.SE3.from_matrix(pred_pose[i])
            self.server.scene.add_camera_frustum(
                f'gt/{i}', fov=1, aspect=1, image=input_frames[i], scale=0.1,
                wxyz=gt_se3.rotation().wxyz, position=gt_se3.translation(),
                color=color_by_id[i])
            
            self.server.scene.add_camera_frustum(
                f'pred/{i}', fov=1, aspect=1, scale=0.1,
                wxyz=pred_se3.rotation().wxyz, position=pred_se3.translation(),
                color=color_by_id[i])
            

viewer = Viewer()
input('Press Enter to exit')