from dataLoaderHelper import *

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
    parser.add_argument("--output_type", type=int,
                        help='output file structure type. 0 for torf. 1 for nerf. 2 for dnerf')
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

    np.save(torf_output_file_path_formatter(getDistanceOutputFolderPath(args, 0), i, cam_id, frame_length), np_distance)
    np.save(torf_output_file_path_formatter(getDistanceMaskOutputFolderPath(args, 0), i, cam_id, frame_length), np_depth_mask)


def torf_save_color(args, i, cam_id, img_i, target_sh, frame_length):
    color = cv2.imread(get_color_img(args, cam_id, img_i))
    color = cv2.resize(color, target_sh[1::-1]) # opencv input dim is (width, height)
    color_mask = (color != 0).any(2)
    np.save(torf_output_file_path_formatter(getColorOutputFolderPath(args, 0), i, cam_id, frame_length), color)
    np.save(torf_output_file_path_formatter(getColorMaskOutputFolderPath(args, 0), i, cam_id, frame_length), color_mask)
    return color

# MARK: dnerf specific
def dnerf_save_depth_and_distance(args, i, cam_id, img_i, k, orig_sh, target_sh, bg, frame_length):
    np_scaled_depth, np_depth_mask, np_distance = make_np_distance_and_mask(args, cam_id, img_i, k, orig_sh, target_sh, bg)

    kwargs = {
        'frame': i
    }
    np.save(os.path.join(getDistanceOutputFolderPath(args, 2, **kwargs), '{}.npy'.format(cam_id)), np_distance)
    np.save(os.path.join(getDistanceMaskOutputFolderPath(args, 2, **kwargs), '{}.npy'.format(cam_id)), np_depth_mask)


def dnerf_save_color(args, i, cam_id, img_i, target_sh, frame_length):
    color = cv2.imread(get_color_img(args, cam_id, img_i))
    color = cv2.resize(color, target_sh[1::-1])  # opencv input dim is (width, height)
    color_mask = (color != 0).any(2)

    kwargs = {
        'frame': i
    }
    cv2.imwrite(os.path.join(getColorOutputFolderPath(args, 2, **kwargs), '{}.png'.format(cam_id)), color)
    np.save(os.path.join(getColorMaskOutputFolderPath(args, 2, **kwargs), '{}.npy'.format(cam_id)), color_mask)
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
        video = cv2.VideoWriter(os.path.join(args.basedir, 'trial_{}/preview/rgb.mp4'.format(args.trial_idx)),
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                15, (512, 384))

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

    np.save(os.path.join(getCamOutputFolderPath(args, 0), 'color_extrinsics.npy'), extrinsics)
    np.save(os.path.join(getCamOutputFolderPath(args, 0), 'tof_extrinsics.npy'), extrinsics)

    np.save(os.path.join(getCamOutputFolderPath(args, 0), 'color_intrinsics.npy'), color_intrinsics)
    np.save(os.path.join(getCamOutputFolderPath(args, 0), 'tof_intrinsics.npy'), color_intrinsics)

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
        video = cv2.VideoWriter(os.path.join(args.basedir, 'trial_{}/preview/rgb.mp4'.format(args.trial_idx)),
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                15, (512, 384))

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
            np.save(os.path.join(getCamOutputFolderPath(args, 2, **kwargs), 'calib.npy'), camera_data)
            if save_video:
                video.write(color_frame)

    if save_video:
        video.release()


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

    if args.output_type == 0:
        # torf data
        make_torf_data(args, **kwargs)

    elif args.output_type == 1:
        make_nerf_data(args, **kwargs)

    elif args.output_type == 2:
        make_dnerf_data(args, **kwargs)
