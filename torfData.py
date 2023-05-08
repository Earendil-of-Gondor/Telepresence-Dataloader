import cv2
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import glob
import torch
import torchvision as tv
from torchvision.utils import save_image

workspace_location = '../../PDFVS_data/setup_1027_wsz/'

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

def getColorOutputFolderPath():
    path = workspace_location+'trial_1/torf_data/color/'
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def getDistanceOutputFolderPath():
    path = workspace_location + 'trial_1/torf_data/distance/'
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def getColorMaskOutputFolderPath():
    path = workspace_location+'trial_1/torf_data/color_mask/'
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def getDistanceMaskOutputFolderPath():
    path = workspace_location + 'trial_1/torf_data/distance_mask/'
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def getCamOutputFolderPath():
    path = workspace_location + 'trial_1/torf_data/cams/'
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def npFormatterIncrementalCamera(basePath, i, cam_id):
    return basePath+'{:04d}.npy'.format(i+len(FRAMES)*CAMERA_IDS.index(cam_id))

# def normalizeImg(low, high, img):
#     imgClip = np.clip(img, low, high)
#     maxVal = np.max(imgClip)
#     minVal = np.min(imgClip)
#     return np.uint8((255.)/(maxVal-minVal)*(imgClip-maxVal)+255.)

def saveDepthAndDistance(i, cam_id, img_i, k):
    depth = torch.load(getDepthFolderPath(cam_id) + '{:05d}.pt'.format(img_i))['depth']
    scaled_depth = torch.nn.functional.interpolate(depth.unsqueeze(0).unsqueeze(0), OUTPUT_SHAPE, mode='nearest').squeeze(0).squeeze(0)
    depth_mask = (scaled_depth != -1)
    np_scaled_depth = scaled_depth.cpu().detach().numpy().astype(np.float32)
    np_depth_mask = depth_mask.cpu().detach().numpy()
    np_distance = computeDistanceMap(np_scaled_depth, k, np_depth_mask)
    np.save(npFormatterIncrementalCamera(getDistanceOutputFolderPath(), i, cam_id),np_distance)
    np.save(npFormatterIncrementalCamera(getDistanceMaskOutputFolderPath(), i, cam_id),np_depth_mask)

def saveColor(i, cam_id, img_i):
    color = cv2.imread(getColorFolderPath(cam_id) + '{:05d}.png'.format(img_i))
    color = cv2.resize(color, (512, 384))
    color_mask = (color != 0).any(2)
    np.save(npFormatterIncrementalCamera(getColorOutputFolderPath(), i, cam_id), color)
    np.save(npFormatterIncrementalCamera(getColorMaskOutputFolderPath(), i, cam_id), color_mask)
    return color

def convertToW2CExtrinsic(tc2w, Rc2w):
    Rw2c = Rc2w.transpose()
    tw2c = -Rc2w.transpose() @ tc2w
    extrinsic = np.eye(4)
    extrinsic[:3,:3] = Rw2c
    extrinsic[:3,[-1]] = tw2c
    return extrinsic

if __name__ == '__main__':
    camera_data = np.load(workspace_location + 'calib.npy', allow_pickle=True).item()

    extrinsics = []

    video = cv2.VideoWriter(workspace_location + 'trial_1/torf_data/rgb.mp4',
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            15, (512, 384))
    for cam_id in CAMERA_IDS:
        # intrinsics
        color_intrinsics = np.eye(4)
        color_intrinsics[:3,:3] = camera_data[cam_id]['K_RGB']
        color_intrinsics /= IMAGE_RATIO
        color_intrinsics[[2, 3],[2, 3]] = 1

        print(color_intrinsics)


        extrinsic = convertToW2CExtrinsic(camera_data[cam_id]['T_to_ref_RGB'], camera_data[cam_id]['R_to_ref_RGB'])

        for i, img_i in enumerate(FRAMES):
            saveDepthAndDistance(i, cam_id, img_i, color_intrinsics)
            video.write(saveColor(i, cam_id, img_i))
            extrinsics.append(extrinsic)

    video.release()

    np.save(getCamOutputFolderPath()+'color_extrinsics.npy', extrinsics)
    np.save(getCamOutputFolderPath()+'tof_extrinsics.npy', extrinsics)

    np.save(getCamOutputFolderPath() + 'color_intrinsics.npy', color_intrinsics)
    np.save(getCamOutputFolderPath() + 'tof_intrinsics.npy', color_intrinsics)




    #
    # intrinsics = np.eye(4)
    # extrinsics = []
    #
    # try:
    #     with open(workspace_location + 'cameras.txt', 'r') as f:
    #         # 1 SIMPLE_RADIAL 1920 1440 1504.0359240533608 960.0 720.0 0.04803028484373421
    #         lines = f.readlines()
    #         camera_str = lines[1]
    #         camera_str_list = camera_str.split()
    #         f = camera_str_list[5]
    #         x = camera_str_list[6]
    #         y = camera_str_list[7]
    #         intrinsics[0,0] = intrinsics[1,1] = f
    #         intrinsics[0,2] = x
    #         intrinsics[1,2] = y
    #
    #     # https://colmap.github.io/format.html#images-txt
    #     with open(workspace_location + 'images.txt', 'r') as f:
    #         lines = f.readlines()
    #         for id, line in enumerate(lines):
    #             if line[0] != '#' and id % 2 == 1:
    #                 # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #                 str_list = line.split()
    #                 camera_id = int(str_list[8])
    #                 camera_serial = str_list[9].split('.')[0]
    #                 quat = np.array([str_list[2], str_list[3], str_list[4], str_list[1]])
    #                 r = R.from_quat(quat).as_matrix()
    #                 t = np.array(str_list[5:8], dtype=float)
    #
    #                 camera_extrinsics = np.eye(4)
    #                 camera_extrinsics[:3,:3] = r
    #                 camera_extrinsics[:3,-1] = t
    #                 extrinsics.append(camera_extrinsics)
    #
    # except FileNotFoundError:
    #     print("The 'docs' directory does not exist")
    #
    # print(intrinsics)
    # np.save('/Users/yuwang/Downloads/torf-main/data/dishwasher_test/cams/color_extrinsics.npy', extrinsics)
    # np.save('/Users/yuwang/Downloads/torf-main/data/dishwasher_test/cams/tof_extrinsics.npy', extrinsics)
