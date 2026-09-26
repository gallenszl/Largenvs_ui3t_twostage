import os
import torch 
# from vggt.models.vggt import VGGT
# from vggt.utils.load_fn import load_and_preprocess_images
from model.vggt.utils.pose_enc import pose_encoding_to_extri_intri
import pose_eval_utils
import json
import numpy as np
from viser import transforms as vtf
from einops import rearrange, repeat
# from vggt.utils.geometry import closed_form_inverse_se3
from pytorch3d.renderer.cameras import PerspectiveCameras
from pose_eval_utils import calculate_auc_np
from tqdm import tqdm
from model.vggt.models.RnG import VGGT4LVSM
import omegaconf
from PIL import Image
from torchvision import transforms as TF


def load_and_preprocess_images(image_path_list, mode='crop'):
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 256

    # First process all images and collect their shapes
    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 8) * 8  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 8) * 8  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 8) * 8

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images



device = "cuda" if torch.cuda.is_available() else "cpu"
# model = VGGT(enable_track=False)
# model.modify_heads()
# ckpt = 'logs/exp001/ckpts/checkpoint_2.pt'
# model.load_state_dict(torch.load(ckpt, map_location='cpu')['model'])

cfg = omegaconf.OmegaConf.load('configs/RnGUP_obj_small_bf16_15k.yaml')
model = VGGT4LVSM(cfg)
ckpt = torch.load('experiments/checkpoints/RnGUP_obj_256P8_b2v3a4Bf16/ckpt_0000000000060000.pt', map_location='cpu')['model']
model.camera_head.to(torch.float32)
model.load_state_dict(ckpt)
model.eval()
model = model.to(device)
model.camera_head.to(torch.float32)


with open('gso_pairs.txt', 'r') as f:
    lines = f.readlines()

name_id_pairs = {i.split(' ')[0]:i.strip().split(' ')[1:] for i in lines}


def build_path_list(name):
    idx_list = name_id_pairs[name][:4]
    idx_list = [int(i) for i in idx_list]
    path_list = [f'/root/mochu_ws/ssd1/gso_render_rv/{name}/{i:03d}.png' for i in idx_list]
    return idx_list, path_list


def get_gt_pose(name, idx_list):
    with open(f'/root/mochu_ws/ssd1/gso_render_rv/{name}/transforms.json', 'r') as f:
        obj_info = json.load(f)
    cam_list = []

    target_first_view = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,-1],[0,0,0,1.]])

    for i, idx in enumerate(idx_list):
        pose = np.array(obj_info['frames'][idx]['transform_matrix'])
        rpy = vtf.SO3.from_matrix(pose[:3,:3]).as_rpy_radians()
        r,p,y = rpy.roll, rpy.pitch, rpy.yaw
        rpy = [r-np.pi/2, -y, p]
        
        pos = pose[:3, 3]
        x,y,z = pos
        xyz = [x, -z, y]
        rot = vtf.SO3.from_rpy_radians(*rpy)
        position = np.array(xyz)
        
        pose1 = np.eye(4)
        pose1[:3,:3] = rot.as_matrix()
        pose1[:3, 3] = position
        
        if i == 0:
            norm_first_view_t = np.linalg.norm(pose1[:3, 3])
            pose1[:3, 3] /= norm_first_view_t
            inv_first_view = np.linalg.inv(pose1)
        else:
            pose1[:3, 3] /= norm_first_view_t

        pose1 = target_first_view @ inv_first_view @ pose1

        cam_list.append(pose1)
        
    cam_list = np.stack(cam_list)
    cam_list = torch.from_numpy(cam_list)#.unsqueeze(0)
    return cam_list


name_list = os.listdir('/root/mochu_ws/ssd1/gso_render_rv')
rError = []
tError = []
for idx, name in enumerate(tqdm(sorted(name_list))):
    # if not idx % 20 == 0:
    #     continue

    print(name)
    idx_list, image_path_list = build_path_list(name)

    images = load_and_preprocess_images(image_path_list).to(device).unsqueeze(0)
    print(f"Preprocessed images shape: {images.shape}, {images.mean(), images.std()}")

    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model.forward_pose_only(images)
    
    print(predictions.shape)

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions, images.shape[-2:])
    # predictions["extrinsic"] = extrinsic
    # predictions["intrinsic"] = intrinsic

    # print('extrinsic.shape:')
    # print(extrinsic.shape)
    
    gt_pose = get_gt_pose(name, idx_list)#.cuda()

    # print(gt_pose.shape)
    
    pred_pose = torch.from_numpy(np.eye(4))#.cuda()
    pred_pose = repeat(pred_pose, 'i j -> 4 i j').contiguous().clone()
    pred_pose[:, :3, :] = extrinsic[0].clone().cpu()
    pred_pose = torch.inverse(pred_pose)
    # pred_pose = closed_form_inverse_se3(pred_pose)
    pred_pose = pred_pose.float()
    # t1 = pred_pose[0, :3, 3].clone()
    # target_pos = torch.tensor([0, 0, -1], dtype=torch.float32, device=pred_pose.device)

    # T = torch.eye(4, dtype=torch.float32, device=pred_pose.device)
    # T[:3, 3] = target_pos - t1
    # pred_pose = T @ pred_pose
    # target_first_view = torch.tensor(
    #     [[1, 0, 0, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 1, -1],
    #     [0, 0, 0, 1]], dtype=torch.float32, device=pred_pose.device
    # )
    # c2w_first = pred_pose[0].clone()

    # norm_first_view_t = torch.norm(c2w_first[:3, 3])
    # c2w_first[:3, 3] /= norm_first_view_t
    # inv_first_view = torch.inverse(c2w_first)

    # for i in range(pred_pose.shape[0]):
    #     pred_pose[i, :3, 3] /= norm_first_view_t
    #     pred_pose[i] = target_first_view @ inv_first_view @ pred_pose[i]

    
    pred_pose = PerspectiveCameras(
        R = pred_pose[:, :3, :3],
        T = pred_pose[:, :3,  3]
    )
    
    gt_pose = PerspectiveCameras(
        R = gt_pose[:, :3, :3],
        T = gt_pose[:, :3,  3]
    )
    
    rel_rangle_deg, rel_tangle_deg = pose_eval_utils.camera_to_rel_deg(pred_pose, gt_pose, 'cpu', 1)
    
    # rel_rangle_deg, rel_tangle_deg = pose_eval_utils.compute_pose_error(pred_pose, gt_pose)
    
    print(rel_rangle_deg)
    print(rel_tangle_deg)
    
    rError.append(rel_rangle_deg)
    tError.append(rel_tangle_deg)
    
    # exit()
    
    
rError = torch.concatenate(rError).numpy()
tError = torch.concatenate(tError).numpy()

print(rError.mean()) # 6.588
print(tError.mean()) # 54.586

Racc_5 = np.mean(rError < 5) * 100
Racc_15 = np.mean(rError < 15) * 100
Racc_30 = np.mean(rError < 30) * 100

Tacc_5 = np.mean(tError < 5) * 100
Tacc_15 = np.mean(tError < 15) * 100
Tacc_30 = np.mean(tError < 30) * 100

Auc_30 = calculate_auc_np(rError, tError, max_threshold=30) * 100

print(Racc_5, Racc_15, Racc_30)
print(Tacc_5, Tacc_15, Tacc_30)

print(Auc_30)