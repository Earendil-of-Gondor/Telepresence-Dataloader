import os

from dataLoaderHelper import *

'''
setup_xxxx_xxx:
    trial_1:
        camid:
            depth:
                00000.pt
                00001.pt
                ...
            color
                00000.png
                00001.png
                ...
    trial_2:
    ...
    background:
        camid:
            depth:
                00000.pt
            color
                00000.png
        camid:
        ...
    calib.npy
'''


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--basedir", type=str,
                        help='base directory')
    parser.add_argument("--trial_idx", type=int,
                        help='trial index')
    parser.add_argument("--structure_type", type=int,
                        help='file structure type. default 0 for separate background and foreground depth')

    parser.add_argument("--outdir", type=str,
                        help='output directory name inside trial folder')
    parser.add_argument("--output_type", type=str,
                        help='seperate by , output file structure type. 0 for torf. 1 for nerf. 2 for dnerf. 3 for tdnerf, '
                             '4 for mnsff')

    parser.add_argument("--frame_start", type=int,
                        help='start index including')
    parser.add_argument("--frame_end", type=int,
                        help='end index including')
    parser.add_argument("--resize_factor", type=int,
                        help='resize by factor')

    # optional argument
    parser.add_argument("--save_video", action='store_true',
                        help='save a video of all frames from all cameras')

    return parser


# MARK: torf specific
def torf_output_file_path_formatter(basePath, i, cam_id, frame_length):
    return os.path.join(basePath, '{:04d}.npy'.format(i + frame_length * CAMERA_IDS.index(cam_id)))


def torf_save_depth_and_distance(args, i, cam_id, img_i, k, orig_sh, target_sh, bg, frame_length):
    np_scaled_depth, np_depth_mask, np_distance = make_np_distance_and_mask(args, cam_id, img_i, k, orig_sh, target_sh,
                                                                            bg)

    np.save(torf_output_file_path_formatter(getDistanceOutputFolderPath(args), i, cam_id, frame_length), np_distance)
    np.save(torf_output_file_path_formatter(getDistanceMaskOutputFolderPath(args), i, cam_id, frame_length), np_depth_mask)


def torf_save_color(args, i, cam_id, img_i, target_sh, frame_length):
    color = cv2.imread(get_color_img(args, cam_id, img_i))
    color = cv2.resize(color, target_sh[1::-1]) # opencv input dim is (width, height)
    color_mask = (color != 0).any(2)
    np.save(torf_output_file_path_formatter(getColorOutputFolderPath(args), i, cam_id, frame_length), color)
    np.save(torf_output_file_path_formatter(getColorMaskOutputFolderPath(args), i, cam_id, frame_length), color_mask)
    return color

# MARK: dnerf specific
def dnerf_save_depth_and_distance(args, i, cam_id, img_i, k, orig_sh, target_sh, bg, frame_length):
    np_scaled_depth, np_depth_mask, np_distance = make_np_distance_and_mask(args, cam_id, img_i, k, orig_sh, target_sh, bg)

    kwargs = {
        'frame': i
    }
    np.save(os.path.join(getDistanceOutputFolderPath(args, **kwargs), '{}.npy'.format(cam_id)), np_distance)
    np.save(os.path.join(getDistanceMaskOutputFolderPath(args, **kwargs), '{}.npy'.format(cam_id)), np_depth_mask)


def dnerf_save_color(args, i, cam_id, img_i, target_sh, frame_length):
    color = cv2.imread(get_color_img(args, cam_id, img_i))
    color = cv2.resize(color, target_sh[1::-1])  # opencv input dim is (width, height)
    color_mask = (color != 0).any(2)

    kwargs = {
        'frame': i
    }
    cv2.imwrite(os.path.join(getColorOutputFolderPath(args, **kwargs), '{}.png'.format(cam_id)), color)
    np.save(os.path.join(getColorMaskOutputFolderPath(args, **kwargs), '{}.npy'.format(cam_id)), color_mask)
    return color

# MARK: tdnerf
def multiview_time_nerf_output_file_path_formatter(basePath, i, cam_id, frame_length):
    return os.path.join(basePath, '{:05d}.npy'.format(i + frame_length * CAMERA_IDS.index(cam_id)))


def multiview_time_nerf_save_depth_and_distance(args, i, cam_id, img_i, k, orig_sh, target_sh, bg, frame_length):
    np_scaled_depth, np_depth_mask, np_distance = make_np_distance_and_mask(args, cam_id, img_i, k, orig_sh, target_sh,
                                                                            bg)
    bds_min = np.amin(np_distance)
    bds_max = np.amax(np_distance)
    bds = np.stack([bds_min, bds_max], 0)
    np.save(multiview_time_nerf_output_file_path_formatter(getDistanceOutputFolderPath(args), i, cam_id, frame_length), np_distance)
    np.save(multiview_time_nerf_output_file_path_formatter(getDistanceMaskOutputFolderPath(args), i, cam_id, frame_length), np_depth_mask)
    return bds


def tdnerf_save_color(args, i, cam_id, img_i, target_sh, frame_length):
    color = cv2.imread(get_color_img(args, cam_id, img_i))
    color = cv2.resize(color, target_sh[1::-1]) # opencv input dim is (width, height)
    color_mask = (color > 0.1).any(2)
    np.save(multiview_time_nerf_output_file_path_formatter(getColorOutputFolderPath(args), i, cam_id, frame_length), color)
    np.save(multiview_time_nerf_output_file_path_formatter(getColorMaskOutputFolderPath(args), i, cam_id, frame_length), color_mask)
    return color

# MARK: mnsff
def mnsff_save_color(args, i, cam_id, img_i, target_sh, frame_length):
    color = cv2.imread(get_color_img(args, cam_id, img_i))
    cv2.imwrite(
        os.path.join(getColorOutputFolderPath(args), '{:05d}.png'.format(i + frame_length * CAMERA_IDS.index(cam_id))),
        color)
    color = cv2.resize(color, target_sh[1::-1]) # opencv input dim is (width, height)
    color_mask = (color != 0).any(2)
    resized_path = os.path.join(get_base_output_path(args), f'images_{target_sh[1]}x{target_sh[0]}')
    if not os.path.exists(resized_path):
        os.makedirs(resized_path)
    cv2.imwrite(os.path.join(resized_path, '{:05d}.png'.format(i + frame_length * CAMERA_IDS.index(cam_id))), color)
    np.save(multiview_time_nerf_output_file_path_formatter(getColorMaskOutputFolderPath(args), i, cam_id, frame_length), color_mask)
    return color

# MARK: data loaders
def make_torf_data(args, factor, frames, frames_length, save_video, img_sh, depth_sh, target_sh):
    '''

    :param args:
    :param factor:
    :param frames:
    :param frames_length:
    :param save_video:
    :param img_sh:
    :param depth_sh:
    :param target_sh:
    :return: -torf_data (all npy files)
                -cams
                    -color_extrinsics
                    -tof_extrinsics
                    -color_intrinsics
                    -tof_intrinsics
                -color: len(frames)*len(cams)
                -distance
                -color_mask
                -distance_mask
    '''
    camera_data = np.load(os.path.join(args.basedir, 'calib.npy'), allow_pickle=True).item()

    extrinsics = []

    if save_video:
        video = cv2.VideoWriter(os.path.join(get_base_output_path(args), 'rgb.mp4'),
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                15, target_sh[:2])

    for cam_id in CAMERA_IDS:
        # intrinsics
        color_intrinsics = make_intrinsic(camera_data[cam_id]['K_RGB'], factor)

        print(color_intrinsics)

        extrinsic = convert_to_w2c_extrinsic(camera_data[cam_id]['T_to_ref_RGB'], camera_data[cam_id]['R_to_ref_RGB'])
        depth_bg = getBackgroundDepthFilePath(args, cam_id)
        depth_bg = torch.load(depth_bg)['depth'].to(device)

        for i, img_i in enumerate(frames):
            torf_save_depth_and_distance(args, i, cam_id, img_i, color_intrinsics, depth_sh, target_sh, depth_bg, frames_length)
            color_frame = torf_save_color(args, i, cam_id, img_i, target_sh, frames_length)
            extrinsics.append(extrinsic)
            if save_video:
                video.write(color_frame)

    if save_video:
        video.release()

    np.save(os.path.join(getCamOutputFolderPath(args), 'color_extrinsics.npy'), extrinsics)
    np.save(os.path.join(getCamOutputFolderPath(args), 'tof_extrinsics.npy'), extrinsics)

    np.save(os.path.join(getCamOutputFolderPath(args), 'color_intrinsics.npy'), color_intrinsics)
    np.save(os.path.join(getCamOutputFolderPath(args), 'tof_intrinsics.npy'), color_intrinsics)

    np.save(os.path.join(get_base_output_path(args), 'test_poses.npy'), np.load('test_poses.npy'))

def make_nerf_data(args, factor, frames, frames_length, save_video, img_sh, depth_sh, target_sh):
    return 0

def make_dnerf_data(args, factor, frames, frames_length, save_video, img_sh, depth_sh, target_sh):
    """

    :param args:
    :param factor:
    :param frames:
    :param frames_length:
    :param save_video:
    :param img_sh:
    :param depth_sh:
    :param target_sh:
    :return: -dnerf_data len(frames)
                -frame_000
                    -calib.npy
                    -images: len(cams)
                    -depths
                    -color_mask
                    -distance_mask
                -frame_001
                ...
    """

    camera_data = np.load(os.path.join(args.basedir, 'calib.npy'), allow_pickle=True).item()

    if save_video:
        video = cv2.VideoWriter(os.path.join(get_base_output_path(args), 'rgb.mp4'),
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                15, target_sh[:2])

    for cam_id in CAMERA_IDS:
        # intrinsics
        color_intrinsics = make_intrinsic(camera_data[cam_id]['K_RGB'], factor)
        camera_data[cam_id]['K_RGB'] = color_intrinsics[:3,:3]

        print(color_intrinsics)

        depth_bg = getBackgroundDepthFilePath(args, cam_id)
        depth_bg = torch.load(depth_bg)['depth'].to(device)

        for i, img_i in enumerate(frames):
            dnerf_save_depth_and_distance(args, i, cam_id, img_i, color_intrinsics, depth_sh, target_sh, depth_bg,
                                         frames_length)
            color_frame = dnerf_save_color(args, i, cam_id, img_i, target_sh, frames_length)
            kwargs = {
                'frame': i
            }
            np.save(os.path.join(getCamOutputFolderPath(args, **kwargs), 'calib.npy'), camera_data)
            if save_video:
                video.write(color_frame)

    if save_video:
        video.release()

def make_tdnerf_data(args, factor, frames, frames_length, save_video, img_sh, depth_sh, target_sh):
    '''

    :param args:
    :param factor:
    :param frames:
    :param frames_length:
    :param save_video:
    :param img_sh:
    :param depth_sh:
    :param target_sh:
    :return: -tdnerf_data (all npy files)
                -poses_bounds.npy
                -images: len(cams) * frames_length
                -depths:
                -color_mask
                -distance_mask

    '''
    camera_data = np.load(os.path.join(args.basedir, 'calib.npy'), allow_pickle=True).item()

    extrinsics = []

    if save_video:
        video = cv2.VideoWriter(os.path.join(get_base_output_path(args), 'rgb.mp4'),
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                15, target_sh[:2])

    poses_bounds = []
    for cam_id in CAMERA_IDS:
        # intrinsics
        color_intrinsics = make_intrinsic(camera_data[cam_id]['K_RGB'], factor)

        print(color_intrinsics)

        depth_bg = getBackgroundDepthFilePath(args, cam_id)
        depth_bg = torch.load(depth_bg)['depth'].to(device)

        for i, img_i in enumerate(frames):
            color_frame = tdnerf_save_color(args, i, cam_id, img_i, target_sh, frames_length)
            bds = multiview_time_nerf_save_depth_and_distance(args, i, cam_id, img_i, color_intrinsics, depth_sh, target_sh, depth_bg, frames_length)
            poses_bounds.append(make_poses_bounds(color_intrinsics, camera_data[cam_id]['T_to_ref_RGB'],
                                                  camera_data[cam_id]['R_to_ref_RGB'], bds))
            if save_video:
                video.write(color_frame)

    poses_bounds = np.stack(poses_bounds, 0)
    # print(poses_bounds[::])
    np.save(os.path.join(getCamOutputFolderPath(args), 'poses_bounds.npy'), poses_bounds)
    if save_video:
        video.release()

def make_mnsff_data(args, factor, frames, frames_length, save_video, img_sh, depth_sh, target_sh):
    '''

    :param args:
    :param factor:
    :param frames:
    :param frames_length:
    :param save_video:
    :param img_sh:
    :param depth_sh:
    :param target_sh:
    :return: -tdnerf_data
                -poses_bounds.npy
                -images(png): len(cams) * frames_length
                -images_123x123(png): len(cams) * frames_length
                -disp(npy):
                -color_mask
                -distance_mask
                -motion_masks(npy)
                -flow_i1(npz)


    '''
    if True:
        camera_data = np.load(os.path.join(args.basedir, 'calib.npy'), allow_pickle=True).item()

        extrinsics = []

        if save_video:
            video = cv2.VideoWriter(os.path.join(get_base_output_path(args), 'rgb.mp4'),
                                    cv2.VideoWriter_fourcc(*'mp4v'),
                                    15, target_sh[:2])

        poses_bounds = []
        for cam_id in CAMERA_IDS:
            # intrinsics
            color_intrinsics = make_intrinsic(camera_data[cam_id]['K_RGB'], factor)

            print(color_intrinsics)

            depth_bg = getBackgroundDepthFilePath(args, cam_id)
            depth_bg = torch.load(depth_bg)['depth'].to(device)

            for i, img_i in enumerate(frames):
                bds = multiview_time_nerf_save_depth_and_distance(args, i, cam_id, img_i, color_intrinsics, depth_sh, target_sh, depth_bg, frames_length)
                color_frame = mnsff_save_color(args, i, cam_id, img_i, target_sh, frames_length)
                poses_bounds.append(make_poses_bounds(color_intrinsics, camera_data[cam_id]['T_to_ref_RGB'],
                                                      camera_data[cam_id]['R_to_ref_RGB'], bds))
                if save_video:
                    video.write(color_frame)

        poses_bounds = np.stack(poses_bounds, 0)
        # print(poses_bounds[::])
        np.save(os.path.join(getCamOutputFolderPath(args), 'poses_bounds.npy'), poses_bounds)

        if save_video:
            video.release()

    # colmap
    # files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]
    #
    # if os.path.exists(os.path.join(getCamOutputFolderPath(args), 'sparse/0')):
    #     files_had = os.listdir(os.path.join(getCamOutputFolderPath(args), 'sparse/0'))
    # else:
    #     files_had = []
    # if not all([f in files_had for f in files_needed]):
    #     print('Need to run COLMAP')
    #     run_colmap(getCamOutputFolderPath(args), 'exhaustive_matcher')
    # else:
    #     print('Don\'t need to run COLMAP')
    #
    # sparse_p0 = os.path.join(getCamOutputFolderPath(args), 'sparse', '0')
    # if os.path.exists(sparse_p0):
    #     print([(subprocess.check_output(['mv', sparse_p0+f'/{f}', os.path.join(getCamOutputFolderPath(args), 'sparse')],
    #                                    universal_newlines=True)) for f in os.listdir(sparse_p0)])
    #     os.rmdir(sparse_p0)

    # image_undistorter_args = [
    #     'colmap', 'image_undistorter',
    #     '--image_path', os.path.join(getCamOutputFolderPath(args), 'images'),
    #     '--input_path', os.path.join(getCamOutputFolderPath(args), 'sparse', '0'),
    #     '--output_path', os.path.join(getCamOutputFolderPath(args), 'dense'),
    #     '--output_type', 'COLMAP',
    #     '--max_image_size', '2000'
    # ]
    #
    # match_output = (subprocess.check_output(image_undistorter_args, universal_newlines=True))
    # print('undistorted', match_output)



    # motion mask and scene flow
    wd = os.getcwd()
    os.chdir(os.path.join(wd, 'nsff_scripts'))

    if not os.path.exists('models/raft-things.pth'):
        download_model_output = (subprocess.check_output(['./download_models.sh'], universal_newlines=True))
        print(download_model_output)

    motion_mask_args = ['python', 'run_flows_video.py',
                        '--model', 'models/raft-things.pth',
                        '--data_path', get_base_output_path(args)]

    motion_mask_output = (subprocess.check_output(motion_mask_args, universal_newlines=True))
    print(motion_mask_output)
    os.chdir(wd)

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    factor = args.resize_factor
    frames = list(range(args.frame_start, args.frame_end + 1))
    frames_length = len(frames)
    save_video = args.save_video
    img_sh = get_img_shape(args)
    depth_sh = get_depth_shape(args)
    target_sh = np.array(img_sh) / factor
    target_sh = tuple(target_sh.astype(int))

    kwargs = {
        'factor': factor,
        'frames': frames,
        'frames_length': frames_length,
        'save_video': save_video,
        'img_sh': img_sh,
        'depth_sh': depth_sh,
        'target_sh': target_sh
    }

    outputs = [int(i) for i in args.output_type.split(',')]

    for output_type in outputs:
        args.output_type = output_type
        if output_type == 0:
            make_torf_data(args, **kwargs)
        elif output_type == 1:
            make_nerf_data(args, **kwargs)
        elif output_type == 2:
            make_dnerf_data(args, **kwargs)
        elif output_type == 3:
            make_tdnerf_data(args, **kwargs)
        elif output_type == 4:
            make_mnsff_data(args, **kwargs)
