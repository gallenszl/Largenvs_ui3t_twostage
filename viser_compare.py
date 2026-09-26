# import utils.tsdf_fusion as fusion 
import importlib
from setup import init_config
import viser 
from viser import transforms as vtf
import torch
import numpy as np 
from einops import rearrange, repeat
from recover_gravity import find_best_x_rotation_zero_yaw
from model.vggt.utils.pose_enc import pose_encoding_to_extri_intri
import random
import os 
import math
from PIL import Image
import cv2
import time
import plotly.express as px
import trimesh
# from data.dataset_gso_ours import GSODataset_ours
import pickle
from torch.utils.data import Dataset
import json
import point_cloud_utils as pcu
import open3d as o3d


bg_img = np.linspace(250, 200, 128).astype(int)
bg_img = repeat(bg_img, 'n -> n 128 3')

white_bg = repeat(np.array([255]*3, dtype=np.uint8), 'n -> 1 1 n')

color_by_id = [
    (38, 70, 83),
    (42, 157, 143),
    (244, 162, 97),
    (193, 56, 22)
]

fixed_location = np.stack([
    np.linspace(0, 2.3*2*np.pi, 13),
    np.linspace(-np.pi*3/4, np.pi*3/4, 13)], -1)


bad_trellis = [15,16,57,86,103,151,153,165,166,173,218,271,291,304,309,413,483,566,579,602,604,865,956]


def pinhole_z_depth_to_xyz(depth, focal=None, H=256, W=256):
    if focal is None:
        fov = 0.6981317007977318
        f = (W/2) / math.tan(fov/2)
    else:
        f = focal

    if isinstance(depth, torch.Tensor):
        depth = depth.numpy()
    if not isinstance(depth, float):
        H, W = depth.shape
        z = depth
    else:
        z = np.ones((H, W), dtype=np.float32) * depth
    y, x = np.mgrid[:H, :W]
    x = x - W // 2
    y = H // 2 - y
    x = x / f * z
    y = - y / f * z
    return np.stack([x, y, z], -1)


with open('trellis_rot.txt', 'r') as f:
    trellis_rot = {line.strip().split(' ')[0]:int(line.strip().split(' ')[1]) for line in f.readlines()}

class GSODataset_ours(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.root_path = self.config.training.val_dataset_cfgs.root_dir

        with open(self.config.training.val_dataset_cfgs.split_file, 'r') as f:
            self.all_object_list = f.readlines()
        self.all_object_list = [i.strip() for i in self.all_object_list]

        self.fov = 0.6981317007977318

        random.seed(0)
        self.rand_idx = [random.sample(range(0, 25), 14) for _ in range(len(self.all_object_list))]

        # rand_idx = np.array(self.rand_idx, dtype=int)
        # np.savetxt('gso_idx.txt', rand_idx, fmt='%i', delimiter=' ')

    def __len__(self):
        return len(self.all_object_list)

    def view_selector(self, idx):
        return self.rand_idx[idx]

    def find_extra_indices(self, indices, num_extra):
        full_indices = set(list(range(25)))
        remain = list(full_indices - set(indices))
        extra = random.sample(remain, num_extra)
        return indices[:4] + extra + indices[4:]

    @staticmethod
    def transform_pose(pose):
        pose = np.array(pose)
        rpy = vtf.SO3.from_matrix(pose[:3,:3]).as_rpy_radians()
        r,p,y = rpy.roll, rpy.pitch, rpy.yaw
        rpy = [r-np.pi/2, -y, p]
        
        pos = pose[:3,3]
        x,y,z = pos
        xyz = [x, -z, y]

        rot = vtf.SO3.from_rpy_radians(*rpy)
        wxyz = rot.wxyz
        position = np.array(xyz)

        pose1 = np.eye(4)
        pose1[:3,:3] = rot.as_matrix()
        pose1[:3,3] = position
        return pose1

    @staticmethod
    def pinhole_z_depth_to_xyz(depth, f, H=512, W=512):
        if isinstance(depth, torch.Tensor):
            depth = depth.numpy()
        if not isinstance(depth, float):
            H, W = depth.shape
            z = depth
        else:
            z = np.ones((H, W), dtype=np.float32) * depth
        y, x = np.mgrid[:H, :W]
        x = x - W // 2
        y = H // 2 - y
        x = x / f * z
        y = - y / f * z
        return np.stack([x, y, z], -1)

    def preprocess_frames(self, object_name, image_indices):
        resize_h = self.config.model.image_tokenizer.image_size
        patch_size = self.config.model.image_tokenizer.patch_size

        images = []
        fxfycxcys = []
        intrinsics = []
        c2ws = []
        point_maps = []
        depth_maps = []

        all_pose_path = os.path.join(self.root_path, object_name, f'transforms.json')
        with open(all_pose_path, 'r') as f:
            all_poses = json.load(f)['frames']

        target_first_view = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,-1],[0,0,0,1.]])

        for v_idx, img_idx in enumerate(image_indices):
            ### image
            cur_image_path = os.path.join(self.root_path, object_name, f'{img_idx:03d}.png')
            image = Image.open(cur_image_path)
            white_bg = Image.new(mode='RGBA', size=image.size, color=(255,)*4)
            image = Image.alpha_composite(white_bg, image)
            image = image.convert("RGB")

            original_image_w, original_image_h = image.size
            
            resize_w = int(resize_h / original_image_h * original_image_w)
            resize_w = int(round(resize_w / patch_size) * patch_size)
            image = image.resize((resize_w, resize_h), resample=Image.LANCZOS)

            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()

            ### intrinsic
            focal_length = (resize_w/2) / math.tan(self.fov/2)
            fxfycxcy = torch.tensor([focal_length, focal_length, resize_w/2, resize_h/2])
            intrinsic_mat = torch.tensor([[focal_length, 0, resize_w/2], [0., focal_length, resize_h/2], [0., 0., 1]])
            images.append(image)
            fxfycxcys.append(fxfycxcy)
            intrinsics.append(intrinsic_mat)


            ### extrinsic
            pose = all_poses[img_idx]['transform_matrix']
            c2w = torch.from_numpy(self.transform_pose(pose)).float()
            if v_idx == 0:
                norm_first_view_t = torch.norm(c2w[:3,3])
                c2w[:3,3] /= norm_first_view_t
                first_view_c2w = c2w
                inv_first_view = torch.inverse(c2w)
            else:
                c2w[:3,3] /= norm_first_view_t
            c2w = target_first_view @ inv_first_view @ c2w

            c2ws.append(c2w)

            ### depth & point map
            cur_depth_path = os.path.join(self.root_path, object_name, f'{img_idx:03d}_depth.png')
            depth = Image.open(cur_depth_path)
            depth = np.array(depth.resize((resize_w, resize_h), resample=Image.NEAREST))
            mask = depth < 65534 # valid mask, 1=fg, 0=bg

            max_depth = all_poses[img_idx]['depth']['max']
            min_depth = all_poses[img_idx]['depth']['min']
            depth_range = max_depth - min_depth

            depth = depth / 65535.0 * depth_range + min_depth
            depth = depth * mask

            # normalize by first view's translation
            depth = torch.from_numpy(depth) / norm_first_view_t

            depth_maps.append(depth.float())

            pt = self.pinhole_z_depth_to_xyz(depth, focal_length)
            pt = torch.from_numpy(pt).float()

            # transform to world coordinate
            h,w,c = pt.shape
            pt = rearrange(pt, 'h w c -> (h w) c')
            ones = torch.ones((h*w, 1), dtype=pt.dtype)
            pt = torch.cat([pt, ones], dim=-1)
            pt = c2w @ pt.T
            pt = pt[:3].T
            pt = rearrange(pt, '(h w) c -> c h w', h=h, w=w)
            pt = pt * torch.from_numpy(mask)[None,:,:].float()

            point_maps.append(pt)

        images = torch.stack(images, dim=0)
        fxfycxcys = torch.stack(fxfycxcys, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        c2ws = torch.stack(c2ws)
        depth_maps = torch.stack(depth_maps, dim=0)
        point_maps = torch.stack(point_maps, dim=0)
        
        return images, fxfycxcys, intrinsics, c2ws, point_maps, depth_maps, first_view_c2w

    def __getitem__(self, idx):
        object_name = self.all_object_list[idx]
        image_indices = self.view_selector(idx)

        image_indices = self.find_extra_indices(image_indices, 11)

        input_images, input_fxfycxcy, input_intrinsics, input_c2ws, point_maps, depth_maps, first_view_c2w = self.preprocess_frames(object_name, image_indices)
        extrinsic = torch.inverse(input_c2ws)

        image_indices = torch.tensor(image_indices).long().unsqueeze(-1)  # [v, 1]
        scene_indices = torch.full_like(image_indices, idx)  # [v, 1]
        indices = torch.cat([image_indices, scene_indices], dim=-1)  # [v, 2]

        return {
            "image": input_images,
            "c2w": input_c2ws,
            "extrinsic": extrinsic,
            "fxfycxcy": input_fxfycxcy,
            "intrinsic": input_intrinsics,
            "index": indices,
            "scene_name": object_name,
            "depth_map": depth_maps,
            "point_map": point_maps,
            "first_view_c2w": first_view_c2w
        }

class Viewer:
    def __init__(self, config, dataset, model):
        self.config = config
        self.dataset = dataset
        self.model = model

        self.server = viser.ViserServer()
        self.server.scene.set_up_direction('-y')
        # self.server.scene.set_background_image(bg_img)
        self._init_ui()
        self.draw_frame()

    def _init_ui(self):
        self.current_batch_id = 0

        with self.server.gui.add_folder('Object'):
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

            self.input_mkdown = self.server.gui.add_markdown('')
                
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
        
    def empty_canvas(self):
        removing_names = ['recon', 'tsdf_mesh', 'accum_pc', 'gen/pcd'] + [f'cam_{i}' for i in range(4)]
        for i in removing_names:
            self.server.scene.remove_by_name(i)

    @staticmethod
    def to_homo(x):
        # x: n * 3
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    
    @staticmethod
    def de_homo(x):
        return x[:, :-1] / x[:, [-1]]

    @staticmethod
    def sample_pts(pc, num):
        idx = pcu.downsample_point_cloud_poisson_disk(pc, radius=-1, target_num_samples=num)
        return pc[idx]

    def draw_gt(self):
        batch = self.dataset[self.current_batch_id]
        object_name = self.dataset.all_object_list[self.current_batch_id]

        self.first_view_c2w = batch['first_view_c2w'].cpu().numpy()[:3,:3]

        num_input_imgs = 25
        all_pts = []
        for idx in range(num_input_imgs):
            mask = batch['depth_map'][idx].cpu().numpy() > 0
            mask = self.erode_mask(mask)
            pts = batch["point_map"][idx].cpu().numpy()
            pts = rearrange(pts, 'c h w -> h w c')
            pts = pts[mask]
            all_pts.append(pts)

        all_pts = np.concatenate(all_pts, axis=0)
        all_pts = self.sample_pts(all_pts, num=8192)
        all_pts = (self.first_view_c2w @ all_pts.T).T

        # np.save(f'gt_pts/{object_name}.npy', all_pts)

        self.server.scene.add_point_cloud('gt_pc', points=all_pts, colors=(50, 50, 50), point_size=0.003, point_shape='circle')

        self.gt_pts = all_pts
        self.gt_std = np.std(self.gt_pts, axis=0)

    def draw_vggt(self):
        batch = self.dataset[self.current_batch_id]
        object_name = self.dataset.all_object_list[self.current_batch_id]
        vggt_pkl = f'/workspace/mochu_workspace/vggt/{object_name}.pkl'
        pred = pickle.load(open(vggt_pkl, 'rb'))
        c2w = pred['pred_pose']
        pred_depth = pred['pred_depth'].squeeze()
        intrinsic = pred['intrinsic']
        focals = intrinsic[0,:,0,0] / 259 * 128

        pcds = [pinhole_z_depth_to_xyz(d,f) for d,f in zip(pred_depth, focals)]

        num_input_imgs = 4
        
        all_pts = []
        tsfm = np.eye(4)
        tsfm[2,-1] = -1
        for idx in range(num_input_imgs):
            mask = self.erode_mask(batch["depth_map"][idx].numpy() > 0)
            pts_iv = pcds[idx][mask]
            pts_iv = self.de_homo((tsfm @ c2w[idx] @ self.to_homo(pts_iv).T).T)
            all_pts.append(pts_iv)

        all_pts = np.concatenate(all_pts, axis=0)
        all_pts = self.sample_pts(all_pts, num=8192)

        all_pts = (self.first_view_c2w @ all_pts.T).T

        # align with gt_pts
        pt_std = np.std(all_pts, axis=0)
        scale = np.mean(self.gt_std / pt_std)
        all_pts = all_pts * scale

        np.save(f'vggt_pts/{object_name}.npy', all_pts)

        self.server.scene.add_point_cloud(f'vggt_pc', points=all_pts, colors=(200, 50, 50),
            point_size=0.003, point_shape='circle')

    @staticmethod
    def rotate_trellis(v, name):
        rot_deg = trellis_rot[name]
        rot = vtf.SO3.from_z_radians(np.deg2rad(rot_deg)).as_matrix()
        rot2 = vtf.SO3.from_x_radians(np.deg2rad(90)).as_matrix()
        return (rot2 @ rot @ v.T).T

    @staticmethod
    def ICP(gt, pred):
        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(pred)

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt)
        init = np.eye(4)

        # Threshold for ICP (max correspondence distance)
        threshold = 0.05

        reg = o3d.pipelines.registration.registration_icp(
            pcd_pred, 
            pcd_gt,
            threshold,
            init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        T = reg.transformation   # 4×4 transform matrix
        print("Estimated ICP transform:\n", T)

        # Apply transform to 'pred'
        pcd_pred_aligned = pcd_pred.transform(T)

        # Convert back to numpy
        pred_aligned = np.asarray(pcd_pred_aligned.points)
        return pred_aligned

    def draw_trellis(self):
        object_name = self.dataset.all_object_list[self.current_batch_id]
        v, f = pcu.load_mesh_vf(f'/workspace/mochu_workspace/trellis_pred/{object_name}.obj')
        fid, bc = pcu.sample_mesh_poisson_disk(v, f, num_samples=8192)
        pts = pcu.interpolate_barycentric_coords(f, fid, bc, v)
        pts = self.rotate_trellis(pts, object_name)

        # align with gt_pts
        pt_std = np.std(pts, axis=0)
        scale = np.mean(self.gt_std / pt_std)
        pts = pts * scale

        pts = self.ICP(self.gt_pts, pts)

        np.save(f'trellis_pts/{object_name}.npy', pts)

        self.server.scene.add_point_cloud(f'trellis_pc', points=pts, colors=(0, 255, 0),
            point_size=0.003, point_shape='circle')


    def draw_frame(self):
        self.empty_canvas()
        self.draw_gt()
        # self.draw_vggt()
        # self.draw_trellis()

        batch = self.dataset[self.current_batch_id]

        with torch.no_grad(), torch.autocast(
            enabled=config.training.use_amp,
            device_type="cuda",
            dtype=amp_dtype_mapping[config.training.amp_dtype]
        ):
            batch = self.dataset[self.current_batch_id]
            batch = {k: v.cuda().unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            pose_enc = self.model.forward_pose_only(batch['image'])
        
            pred_ext, intrinsic = pose_encoding_to_extri_intri(pose_enc, (256,256))
        
        mask = batch['depth_map'].cpu().numpy() > 0

        # pred_ext = pred_ext.squeeze(0)
        # ext = torch.eye(4).unsqueeze(0).repeat(pred_ext.shape[0], 1,1)

        # ext[:,:3,:] = pred_ext
        # c2w = torch.inverse(ext)
        # c2w = c2w.numpy()

        c2w = batch['c2w'].squeeze()

        all_pts = []
        num_input_imgs = 4
        for idx in range(num_input_imgs):
            img = batch["image"][0, idx].cpu().numpy()
            img = (rearrange(img, 'c h w -> h w c') * 255).astype(np.uint8)
            # wxyz_xyz = vtf.SE3.from_matrix(c2w[idx]).wxyz_xyz

            # self.server.scene.add_camera_frustum(
            #     f'RnG/cam_{idx}', wxyz=wxyz_xyz[:4], position=wxyz_xyz[4:],
            #     fov=1, aspect=1, scale=0.2, color=color_by_id[idx], line_width=6)
            
            target_img, pts, pts_conf, _mask = self.forward_single_view(c2w[idx].cpu().numpy(), use_conf_mask=True)
            mask_ = self.erode_mask(mask[0, idx])
            pts_iv = pts[mask_]
            all_pts.append(pts_iv)
        
        for idx in range(4, 25):
            target_img, pts, pts_conf, _mask = self.forward_single_view(c2w[idx].cpu().numpy(), use_conf_mask=True)
            mask_ = self.erode_mask(_mask, kernel=13)
            pts_iv = pts[mask_]
            all_pts.append(pts_iv)

        all_pts = np.concatenate(all_pts, axis=0)
        all_pts = (self.first_view_c2w @ all_pts.T).T

        all_pts = self.sample_pts(all_pts, num=8192)

        object_name = self.dataset.all_object_list[self.current_batch_id]
        np.save(f'RnG_pts/{object_name}.npy', all_pts)

        self.server.scene.add_point_cloud(f'RnG_pts', points=all_pts, colors=(50, 50, 200),
            point_size=0.003, point_shape='circle')
        
    def forward_single_view(self, target_c2w, use_conf_mask=False):
        # target_c2w = self.gravity_rectification @ target_c2w
        target_c2w = rearrange(torch.from_numpy(target_c2w), 'i j -> 1 1 i j').float().cuda()

        with torch.no_grad(), torch.autocast(
            enabled=config.training.use_amp,
            device_type="cuda",
            dtype=amp_dtype_mapping[config.training.amp_dtype],
        ):
            batch = self.dataset[self.current_batch_id]
            batch = {k: v.cuda()[:4].unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            torch.cuda.synchronize()
            time1 = time.time()
            render_pack = self.model.forward_rendering_using_kv_cache(target_c2w)
            torch.cuda.synchronize()
            print('forward time:', time.time()-time1)
            target_img = render_pack.render
            target_img = rearrange(target_img.float(), '1 1 c h w -> h w c').detach().cpu().numpy()
            target_img = (target_img * 255).astype(np.uint8)

            pts = render_pack.points
            pts_conf = render_pack.points_conf

            pts = rearrange(pts, '1 1 c h w -> h w c').detach().cpu().numpy()
            pts_conf = rearrange(pts_conf, '1 1 1 h w -> h w').detach().cpu().numpy()

            # conf_thresh = np.quantile(pts_conf, 0.1)
            # mask = np.logical_and((pts_conf > conf_thresh), (np.abs(pts).sum(-1) > 1e-2))
        
            bg_color = np.array([[[255,255,255]]])
            bg_mask = np.sum(np.abs(target_img-bg_color), -1) > 12

            if use_conf_mask:
                conf_thresh = np.quantile(pts_conf[bg_mask], 0.03)
                mask = np.logical_and((pts_conf > conf_thresh), (np.abs(pts).sum(-1) > 1e-2), bg_mask)
            else:
                mask = np.logical_and((np.abs(pts).sum(-1) > 1e-2), bg_mask)

        return target_img, pts, pts_conf, mask
    
    @staticmethod
    def erode_mask(mask, kernel=3):
        kernel = np.ones((kernel, kernel), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        return mask.astype(bool)

    def encode_img_path_to_mkdown(self, image_paths):
        rows = ['Input Images:']
        num_cols = 4
        for i in range(0, len(image_paths), num_cols):
            batch = image_paths[i:i + num_cols]
            row = " | ".join(f"![image]({path})" for path in batch)
            sep = " | ".join(["---"] * len(batch))
            rows.append(row)
            rows.append(sep)
        
        return "\n".join(rows)
    
    def autorun(self):
        for i in bad_trellis:
            self.slider.value = i

config = init_config()
module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config, use_kv_cache=True).cuda()
model.load_ckpt(config.training.checkpoint_dir)
model.eval()

# model = None

torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype_mapping = {
    "fp16": torch.float16, 
    "bf16": torch.bfloat16, 
    "fp32": torch.float32, 
    'tf32': torch.float32
}

# dataset = GSOEvalDataset(config)

dataset = GSODataset_ours(config)

viewer = Viewer(config, dataset, model)
viewer.autorun()

# input('Press Enter to exit')
exit()