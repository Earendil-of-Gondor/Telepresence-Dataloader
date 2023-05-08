import cv2
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import glob
import torch
import torchvision as tv
from torchvision.utils import save_image
from llff.poses.pose_utils import gen_poses

workspace_location = '../../data/PDFVS_data/setup_1027_wsz/'

OUT_FOLDER_NAME = 'trial_1'

CAMERA_IDS = ['000228212212', '000655594512', '000665414412', '000738314412']

IMAGE_RATIO = 4

OUTPUT_SHAPE = (384, 512)

FRAMES = list(range(350,381))

def computeDistanceMap(depthMap, k, depth_mask):
    cx = k[0,2]
    cy = k[1,2]
    f = k[0,0]
    distanceMap = np.zeros(OUTPUT_SHAPE)

    for i in range(len(depthMap)):
        for j in range(len(depthMap[0])):
            if depth_mask[i,j]:
                distance = depthToDistanceSinglePoint(i,j,depthMap[i,j],cx,cy,f)
                distanceMap[i,j] = distance

    return distanceMap


def depthToDistanceSinglePoint(u,v,d,cx,cy,f):
    factor = d / f
    return np.linalg.norm(np.array([factor*(u-cx), factor*(v-cy), d]))


def getColorFolderPath(camId):
    return workspace_location+'trial_1/'+camId+'/color/'

def getDepthFolderPath(camId):
    return workspace_location+'trial_1/'+camId+'/depth/'

def getSceneOutputFolderPath(i):
    return workspace_location+'trial_1/dsnerf_data/{:03d}/'.format(i)

def getColorOutputFolderPath(i):
    path = getSceneOutputFolderPath(i)+'images/'
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def getColorMaskOutputFolderPath(i):
    path = getSceneOutputFolderPath(i)+'masks/'
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def getDepthOutputFolderPath(i):
    path = getSceneOutputFolderPath(i)+'depth/'
    if not os.path.exists(path):
        os.makedirs(path)
    return path




# def npFormatterIncrementalCamera(basePath, i, cam_id):
#     return basePath+'{:04d}.npy'.format(i+len(FRAMES)*CAMERA_IDS.index(cam_id))

# def normalizeImg(low, high, img):
#     imgClip = np.clip(img, low, high)
#     maxVal = np.max(imgClip)
#     minVal = np.min(imgClip)
#     return np.uint8((255.)/(maxVal-minVal)*(imgClip-maxVal)+255.)

def saveDepth(i, cam_id, img_i):
    depth = torch.load(getDepthFolderPath(cam_id) + '{:05d}.pt'.format(img_i))['depth']
    scaled_depth = torch.nn.functional.interpolate(depth.unsqueeze(0).unsqueeze(0), OUTPUT_SHAPE, mode='nearest').squeeze(0).squeeze(0)
    np_scaled_depth = scaled_depth.cpu().detach().numpy().astype(np.float32)
    np.save(getDepthOutputFolderPath(i)+'{:03d}.npy'.format(CAMERA_IDS.index(cam_id)), np_scaled_depth)

def saveColor(i, cam_id, img_i):
    color = cv2.imread(getColorFolderPath(cam_id) + '{:05d}.png'.format(img_i))
    color = cv2.resize(color, (512, 384))
    color_mask = (color != 0).any(2) * 255
    cv2.imwrite(getColorOutputFolderPath(i)+'{:03d}.png'.format(CAMERA_IDS.index(cam_id)), color)
    cv2.imwrite(getColorMaskOutputFolderPath(i)+'{:03d}.png'.format(CAMERA_IDS.index(cam_id)), color_mask)
    return color

def convertToLLFFPose(tc2w, Rc2w, f):
    extrinsic = np.eye(4)
    extrinsic[:3,:3] = Rc2w
    extrinsic[:3,[-1]] = tc2w

    hwf = np.array([OUTPUT_SHAPE[0],OUTPUT_SHAPE[1],f]).reshape([3,1])
    return extrinsic, hwf

def buildPosesBounds(extrinsics, hwfs):
    poses = extrinsics[:, :3, :4].transpose([1, 2, 0])
    poses = np.concatenate([poses, np.stack(hwfs, -1)], 1)

    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]],
                           1)

    save_arr = []
    for i in range(poses.shape[-1]):
        save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([0.1, 5])], 0))
    save_arr = np.array(save_arr)

    return save_arr


if __name__ == '__main__':
    camera_data = np.load(workspace_location + 'calib.npy', allow_pickle=True).item()
    # fs = glob.glob(workspace_location+'trial_1/dsnerf_data/colmap_ref/images/*')
    # for f in fs:
    #     print(f)
    #     color = cv2.imread(f)
    #     color = cv2.resize(color, (512, 384))
    #     cv2.imwrite(f, color)

    # video = cv2.VideoWriter(workspace_location + 'trial_1/torf_data/rgb.mp4',
    #                         cv2.VideoWriter_fourcc(*'mp4v'),
    #                         15, (512, 384))
    poses_bounds = np.load(workspace_location + 'trial_1/poses_bounds.npy')
    extrinsics = []
    hwfs = []

    for cam_id in CAMERA_IDS:
        # # poses_bounds
        color_intrinsics = camera_data[cam_id]['K_RGB']
        color_intrinsics /= IMAGE_RATIO
        color_intrinsics[2,2] = 1

        fx = color_intrinsics[0,0]
        fy = color_intrinsics[1,1]

        f = (fx+fy)/2

        print(color_intrinsics)

        extrinsic, hwf = convertToLLFFPose(camera_data[cam_id]['T_to_ref_RGB'], camera_data[cam_id]['R_to_ref_RGB'], f)
        extrinsics.append(extrinsic)
        hwfs.append(hwf)

        for i, img_i in enumerate(FRAMES):
            saveDepth(i, cam_id, img_i)
            saveColor(i, cam_id, img_i)
            np.save(getSceneOutputFolderPath(i)+'poses_bounds.npy', poses_bounds)
            # video.write(saveColor(i, cam_id, img_i))

    extrinsics = np.stack(extrinsics, axis=0)
    hwfs = np.stack(hwfs, axis=0)
    poses_bounds = buildPosesBounds(extrinsics, hwfs)
    np.save(workspace_location + 'trial_1/poses_bounds.npy',poses_bounds)
    print(np.load(workspace_location + 'trial_1/poses_bounds_col.npy'))
    print(np.load(workspace_location + 'trial_1/poses_bounds.npy'))


    # video.release()
    #
    # np.save(getCamOutputFolderPath()+'color_extrinsics.npy', extrinsics)
    # np.save(getCamOutputFolderPath()+'tof_extrinsics.npy', extrinsics)
    #
    # np.save(getCamOutputFolderPath() + 'color_intrinsics.npy', color_intrinsics)
    # np.save(getCamOutputFolderPath() + 'tof_intrinsics.npy', color_intrinsics)



