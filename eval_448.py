#%%
import imageio 
import numpy as np 
import os
import cv2
from tqdm import trange
from matplotlib import pyplot as plt

folder_448 = 'experiments/evaluation/test_448_LinHead'
folder_256 = 'experiments/evaluation/test_256_LinHead'

psnr_448_list = []
psnr_256_list = []

# for idx in trange(1030):
#     obj_name = f'{idx:06d}'
#     img_448_path = f'{folder_448}/{obj_name}/gt_vs_pred.png'
#     img_448 = imageio.imread(img_448_path)
#     gt_448 = img_448[:448]
#     pred_448 = img_448[448:]

#     img_256_path = f'{folder_256}/{obj_name}/gt_vs_pred.png'
#     img_256 = imageio.imread(img_256_path)
#     pred_256 = img_256[256:]
#     pred_256 = cv2.resize(pred_256, (4480, 448), interpolation=cv2.INTER_LINEAR)

#     gt_448 = gt_448.astype(np.float32) / 255.
#     pred_448 = pred_448.astype(np.float32) / 255.
#     pred_256 = pred_256.astype(np.float32) / 255.

#     psnr_448 = -10 * np.log10(np.mean((gt_448 - pred_448) ** 2))
#     psnr_256 = -10 * np.log10(np.mean((gt_448 - pred_256) ** 2))
    
#     psnr_448_list.append(psnr_448)
#     psnr_256_list.append(psnr_256)

# print('PSNR of 448x448 images:', np.mean(psnr_448_list))
# print('PSNR of 256x256 images:', np.mean(psnr_256_list))

# for idx in trange(1030):
for idx in trange(1):
    obj_name = f'{idx:06d}'
    img_448_path = f'{folder_448}/{obj_name}/gt_vs_pred.png'
    img_448 = imageio.imread(img_448_path)
    pred_448 = img_448[448:]

    img_256_path = f'{folder_256}/{obj_name}/gt_vs_pred.png'
    img_256 = imageio.imread(img_256_path)
    pred_256 = img_256[256:]
    gt_256 = img_256[:256]

    pred_448 = cv2.resize(pred_448, (2560, 256), interpolation=cv2.INTER_LINEAR)
    pred_448 = pred_448.astype(np.float32) / 255.
    pred_256 = pred_256.astype(np.float32) / 255.
    gt_256 = gt_256.astype(np.float32) / 255.

    plt.imshow(pred_448)
    plt.show()

    plt.imshow(pred_256)
    plt.show()

    plt.imshow(gt_256)
    plt.show()

    psnr_448 = -10 * np.log10(np.mean((gt_256 - pred_448) ** 2))
    psnr_256 = -10 * np.log10(np.mean((gt_256 - pred_256) ** 2))
    
    psnr_448_list.append(psnr_448)
    psnr_256_list.append(psnr_256)

print('PSNR of 448x448 images:', np.mean(psnr_448_list))
print('PSNR of 256x256 images:', np.mean(psnr_256_list))
# %%
