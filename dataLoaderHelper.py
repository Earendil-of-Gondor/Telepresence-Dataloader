import torch.nn.functional as F
import os
import shutil
import cv2
import numpy as np
import torch
import subprocess
from llff.poses.pose_utils import run_colmap

CAMERA_IDS = ['000228212212', '000597713512', '000655594512', '000665414412', '000738314412', '000907513512']
TEST_CAMERA_IDX = ['000597713512', '000907513512']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# MARK: file source
def get_trial_folder(args):
    return os.path.join(args.basedir, 'trial_{}'.format(args.trial_idx))


def get_camera_folder(args, cam_id):
    return os.path.join(get_trial_folder(args), '{}'.format(cam_id))


def getColorFolderPath(args, camId):
    return os.path.join(get_camera_folder(args, camId), 'color')


def get_color_img(args, cam_id, img_i):
    return os.path.join(getColorFolderPath(args, cam_id), '{:05d}.png'.format(img_i))


def getDepthFolderPath(args, camId):
    return os.path.join(get_camera_folder(args, camId), 'depth')


def get_depth_img(args, cam_id, img_i):
    return os.path.join(getDepthFolderPath(args, cam_id), '{:05d}.pt'.format(img_i))


def getBackgroundDepthFilePath(args, camId):
    return os.path.join(args.basedir, 'background', camId, 'depth', '00000.pt')


def get_img_shape(args):
    img = cv2.imread(get_color_img(args, CAMERA_IDS[0], args.frame_start))
    return img.shape


def get_depth_shape(args):
    img = torch.load(getBackgroundDepthFilePath(args, CAMERA_IDS[0]))['depth']
    return img.shape


# file des
def concateStringRemoveBeginningDup(str1, str2):
    if str1[-1] != '_': str1 = str1 + '_'
    if str2.find(str1) > 0:
        return str2
    else:
        return str1 + str2


def get_base_output_path(args):
    output_type = args.output_type
    if output_type == 0:
        path = os.path.join(get_trial_folder(args), concateStringRemoveBeginningDup('torf_data', args.outdir))
    elif output_type == 1:
        path = os.path.join(get_trial_folder(args), concateStringRemoveBeginningDup('nerf_data', args.outdir))
    elif output_type == 2:
        path = os.path.join(get_trial_folder(args), concateStringRemoveBeginningDup('dnerf_data', args.outdir))
    elif output_type == 3:
        path = os.path.join(get_trial_folder(args), concateStringRemoveBeginningDup('tdnerf_data', args.outdir))
    elif output_type == 4:
        path = os.path.join(get_trial_folder(args), concateStringRemoveBeginningDup('mnsff_data', args.outdir))
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def getCamOutputFolderPath(args, **kwargs):
    output_type = args.output_type
    if output_type == 0:
        path = os.path.join(get_base_output_path(args), 'cams')
    elif output_type == 1:
        path = os.path.join(get_base_output_path(args), 'color')
    elif output_type == 2:
        path = os.path.join(get_base_output_path(args), 'frame_{:03d}'.format(kwargs['frame']))
    elif output_type == 3:
        path = os.path.join(get_base_output_path(args))
    elif output_type == 4:
        path = os.path.join(get_base_output_path(args))
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def getColorOutputFolderPath(args, **kwargs):
    output_type = args.output_type
    if output_type == 0:
        path = os.path.join(get_base_output_path(args), 'color')
    elif output_type == 1:
        path = os.path.join(get_base_output_path(args), 'color')
    elif output_type == 2:
        path = os.path.join(get_base_output_path(args),
                            'frame_{:03d}'.format(kwargs['frame']), 'images')
    elif output_type == 3:
        path = os.path.join(get_base_output_path(args), 'images')
    elif output_type == 4:
        path = os.path.join(get_base_output_path(args), 'images')
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def getDistanceOutputFolderPath(args, **kwargs):
    output_type = args.output_type
    if output_type == 0:
        path = os.path.join(get_base_output_path(args), 'distance')
    elif output_type == 1:
        path = os.path.join(get_base_output_path(args), 'color')
    elif output_type == 2:
        path = os.path.join(get_base_output_path(args),
                            'frame_{:03d}'.format(kwargs['frame']), 'depths')
    elif output_type == 3:
        path = os.path.join(get_base_output_path(args), 'depths')
    elif output_type == 4:
        path = os.path.join(get_base_output_path(args), 'disp')
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def getColorMaskOutputFolderPath(args, **kwargs):
    output_type = args.output_type
    if output_type == 0:
        path = os.path.join(get_base_output_path(args), 'color_mask')
    elif output_type == 1:
        path = os.path.join(get_base_output_path(args), 'color')
    elif output_type == 2:
        path = os.path.join(get_base_output_path(args),
                            'frame_{:03d}'.format(kwargs['frame']), 'color_mask')
    elif output_type == 3:
        path = os.path.join(get_base_output_path(args), 'color_mask')
    elif output_type == 4:
        path = os.path.join(get_base_output_path(args), 'color_mask')
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def getDistanceMaskOutputFolderPath(args, **kwargs):
    output_type = args.output_type
    if output_type == 0:
        path = os.path.join(get_base_output_path(args), 'distance_mask')
    elif output_type == 1:
        path = os.path.join(get_base_output_path(args), 'color')
    elif output_type == 2:
        path = os.path.join(get_base_output_path(args),
                            'frame_{:03d}'.format(kwargs['frame']), 'distance_mask')
    elif output_type == 3:
        path = os.path.join(get_base_output_path(args), 'distance_mask')
    elif output_type == 4:
        path = os.path.join(get_base_output_path(args), 'distance_mask')
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# MARK: camera and pose
def convert_to_w2c_extrinsic(tc2w, Rc2w):
    Rw2c = Rc2w.transpose()
    tw2c = -Rc2w.transpose() @ tc2w
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = Rw2c
    extrinsic[:3, [-1]] = tw2c
    return extrinsic


def make_intrinsic(K, factor):
    color_intrinsics = np.eye(4)
    color_intrinsics[:3, :3] = K
    color_intrinsics /= factor
    color_intrinsics[[2, 3], [2, 3]] = 1
    return color_intrinsics


def make_poses_bounds(color_intrinsics, tc2w, Rc2w, bds):
    # poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    pose = np.zeros((3, 5))
    pose[:3, :3] = Rc2w
    pose[:3, [3]] = tc2w
    pose[2, 4] = color_intrinsics[0, 0]
    pose = np.concatenate([pose[:, 1:2], pose[:, 0:1], -pose[:, 2:3], pose[:, 3:]], 1)
    pose = pose.flatten()
    pose = np.concatenate([pose, bds], 0)
    return pose


# MARK: depth processing
def load_sparse_depth(d_pix_in, d_pix_indices_in, hw):
    d_pix = d_pix_in.to(device)
    d_pix_indices = d_pix_indices_in
    cur_depth = torch.zeros(hw).to(device)
    cur_s = 0
    cur_e = 0
    n_added = 0
    for row in range(cur_depth.shape[0]):
        if d_pix_indices[row] is not None:
            idx_cur_row = d_pix_indices[row].long()
            cur_s = cur_e
            cur_e += idx_cur_row.shape[0]
            cur_depth[row][idx_cur_row] = d_pix[cur_s:cur_e]
            n_added += idx_cur_row.shape[0]
    return cur_depth


def compute_distance_map(depth_map, k, depth_mask):
    cx = k[0, 2]
    cy = k[1, 2]
    f = k[0, 0]
    distance_map = np.zeros_like(depth_map)

    for i in range(len(depth_map)):
        for j in range(len(depth_map[0])):
            if depth_mask[i, j]:
                distance = depth_to_distance_single_point(i, j, depth_map[i, j], cx, cy, f)
                distance_map[i, j] = distance

    return distance_map


def depth_to_distance_single_point(u, v, d, cx, cy, f):
    factor = d / f
    return np.linalg.norm(np.array([factor * (u - cx), factor * (v - cy), d]))


def compute_depth(args, cam_id, img_i, orig_sh):
    data = torch.load(get_depth_img(args, cam_id, img_i))
    cur_d_pixs = data['foreground_depth'].float()
    dmap_idx = data['foreground_depth_indices']
    dmap = load_sparse_depth(cur_d_pixs, dmap_idx, orig_sh[:2])
    return dmap


def compute_merged_depth_scaled(args, cam_id, img_i, orig_sh, target_sh, bg):
    dmap = compute_depth(args, cam_id, img_i, orig_sh)

    bg_mask = (dmap > 0)  # true means we have dynamic depth, don't use bg depth
    bg[bg_mask] = 0
    depth_merged = dmap + bg

    scaled_depth = F.interpolate(depth_merged[None, None], target_sh[:2], mode='nearest')[0][0]

    return scaled_depth


def make_np_distance_and_mask(args, cam_id, img_i, k, orig_sh, target_sh, bg):
    if cam_id in TEST_CAMERA_IDX:
        return np.zeros(target_sh[:2]), np.zeros(target_sh[:2]).astype(bool), np.zeros(target_sh[:2])

    scaled_depth = compute_merged_depth_scaled(args, cam_id, img_i, orig_sh, target_sh, bg)

    depth_mask = (scaled_depth > 1e-5)  # true means there is depth

    np_scaled_depth = scaled_depth.cpu().detach().numpy().astype(np.float32)
    np_depth_mask = depth_mask.cpu().detach().numpy()
    np_distance = compute_distance_map(np_scaled_depth, k, np_depth_mask)

    return np_scaled_depth, np_depth_mask, np_distance
