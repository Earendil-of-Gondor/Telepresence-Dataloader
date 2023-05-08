import glob
import numpy as np
import cv2
import os
from skimage.metrics import structural_similarity
# import lpips
# from PerceptualSimilarity
import torch
import perceptual_loss
#mean psnr: 23.169714, SSIM: 0.798634, LPIPS: 0.117989 660-690 right
#mean psnr: 22.654110, SSIM: 0.756185, LPIPS: 0.158454 660-690 left
def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--gt_folder", type=str,
                        help='ground truth directory, npy file')
    parser.add_argument("--mask_folder", type=str,
                        help='mask directory, npy file')
    parser.add_argument("--test_folder", type=str,
                        help='rendered image directory, png file')
    parser.add_argument("--start_frame", type=int, default=0,
                        help='31')
    parser.add_argument("--end_frame", type=int, default=None,
                        help='62')
    parser.add_argument("--test_start_frame", type=int, default=0,
                        help='0')
    parser.add_argument("--test_end_frame", type=int, default=None,
                        help='31')
    parser.add_argument("--default_setting", action='store_true',
                        help='use default camera and total image to find frames')
    parser.add_argument("--n_cam", type=int, default=6,
                        help='6')
    return parser

def im2tensor(image, imtype=np.uint8, cent=1., factor=1./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


def calculate_metrics(args):
    model = perceptual_loss.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, version=0.1)

    import matplotlib.pyplot as plt

    # gt_folder = '../../../PDFVS_data/setup_1027_wsz/trial_1/torf_data/color/'
    # mask_folder = '../../../PDFVS_data/setup_1027_wsz/trial_1/torf_data/color_mask/'
    # test_folder = '../../torf-main/logs/pdfvs_torf_data_d5tw5r512i0S/perceptual_loss/rgb_raw/'
    start_frame = args.start_frame
    end_frame = args.end_frame
    test_start_frame = args.test_start_frame
    test_end_frame = args.test_end_frame
    gt_img_fnames = [os.path.join(args.gt_folder, f) for f in sorted(os.listdir(args.gt_folder)) if
                   f.endswith('npy')][start_frame:end_frame]
    mask_fnames = [os.path.join(args.mask_folder, f) for f in sorted(os.listdir(args.mask_folder)) if
                   f.endswith('npy')][start_frame:end_frame]
    test_imgs_fnames = [os.path.join(args.test_folder, f) for f in sorted(os.listdir(args.test_folder)) if
                   f.endswith('png')][test_start_frame:test_end_frame]

    if args.default_setting:
        test_cam_idx = [1,5]
        n_frames = int(len(gt_img_fnames) / args.n_cam)
        gt_img_fnames = gt_img_fnames[n_frames:2*n_frames]+ gt_img_fnames[5*n_frames:6*n_frames]
        mask_fnames = mask_fnames[n_frames:2*n_frames] + mask_fnames[5*n_frames:6*n_frames]
        test_imgs_fnames = test_imgs_fnames


    gt_imgs = [np.load(f) for f in gt_img_fnames]
    masks = [np.load(f) for f in mask_fnames]
    test_imgs = [cv2.imread(f) for f in test_imgs_fnames]

    if len(gt_img_fnames) != len(mask_fnames) or len(gt_img_fnames) != len(test_imgs_fnames):
        print('WRONG DEMENSION', f'gt_img_fnames: {len(gt_img_fnames)}, mask_fnames: {len(mask_fnames)}'
                                 f'test_imgs_fnames: {len(test_imgs_fnames)}')

    out = {}
    inv_psnr_list = []
    inv_ssim_list = []
    inv_LPIPS_list = []
    for gt_img, mask, test_img in zip(gt_imgs, masks, test_imgs):
        test_img[~mask] = 0

        PSNR = cv2.PSNR(gt_img, test_img)
        SSIM = structural_similarity(gt_img, test_img, multichannel=True)
        test_img = im2tensor(test_img).cuda() / 255
        gt_img = im2tensor(gt_img).cuda() / 255

        with torch.no_grad():
            lpips_loss = model.forward(gt_img, test_img).item()

        print(f"cur_psnr: {PSNR:.6f}, SSIM: {SSIM:.6f}, LPIPS: {lpips_loss:.6f}")
        inv_psnr_list.append(PSNR)
        inv_ssim_list.append(SSIM)
        inv_LPIPS_list.append(lpips_loss)

    print(f"mean psnr: {np.mean(inv_psnr_list):.6f}, SSIM: {np.mean(inv_ssim_list):.6f}, LPIPS: {np.mean(inv_LPIPS_list):.6f}")
    print(f"{np.mean(inv_psnr_list):.2f} / {np.mean(inv_ssim_list):.2f} / {np.mean(inv_LPIPS_list):.2f}")
    print(f"{np.mean(inv_psnr_list):.4f} / {np.mean(inv_ssim_list):.4f} / {np.mean(inv_LPIPS_list):.4f}")
    out['nerf_psnr_list'] = inv_psnr_list
    out['nerf_LPIPS_list'] = inv_LPIPS_list
    # mean psnr: 20.874201, LPIPS: 0.193876
    # mean psnr: 23.461313, LPIPS: 0.166307
    #
    # if DO_NGP:
    #     ngp_psnr_list = []
    #     for i, (image, frame) in enumerate(zip(ngp_images, ngp_frames)):
    #         print(f"{i}/{len(ngp_images)}")
    #         a = cv2.imread(image)
    #         b = cv2.imread(frame)
    #         cur_psnr = cv2.PSNR(a, b)
    #         ngp_psnr_list.append(cur_psnr)
    #     out['ngp_psnr_list'] = ngp_psnr_list
    #
    # if DO_PSNR_LPIPS_ONE_PLOT:
    #     fig, ax1 = plt.subplots()
    #     ax1.set_xlabel('frame number', fontsize=14, fontweight="bold")
    #     ax1.set_ylabel('PSNR', fontsize=14, fontweight="bold")
    #     ax1.plot(list(range(300)), inv_psnr_list)
    #     ax1.tick_params(axis='y')
    #
    #     # Adding Twin Axes to plot using dataset_2
    #     ax2 = ax1.twinx()
    #
    #     color = 'tab:green'
    #     ax2.set_ylabel('LPIPS', color=color, fontsize=14, fontweight="bold")
    #     ax2.plot(list(range(300)), inv_LPIPS_list, color=color)
    #     ax2.tick_params(axis='y', labelcolor=color)
    #
    #     # Adding title
    #     plt.title(f'mean PSNR: {np.mean(inv_psnr_list):.4f}, LPIPS: {np.mean(inv_LPIPS_list):.4f}',
    #               fontweight="bold", fontsize=16)
    #     plt.show()
    #
    # if DO_NGP and DO_MERGE_GRAPH:
    #     title_str = f'INV mean: {np.mean(inv_psnr_list):.3f}, median{np.median(inv_psnr_list):.3f}\n'+\
    #                 f'NGP mean: {np.mean(ngp_psnr_list):.3f}, median{np.median(ngp_psnr_list):.3f}\n'
    #     plt.plot(list(range(len(inv_psnr_list))), inv_psnr_list, color='g', label='INV')
    #     plt.plot(list(range(len(ngp_psnr_list))), ngp_psnr_list, color='r', label='NGP + Incre Xfer')
    #     plt.xlabel('frame #')
    #     plt.ylabel('PSNR')
    #     plt.ylim([15, 35])
    #     plt.title(title_str)
    #     plt.legend()
    #     # function to show the plot
    #     plt.show()

    # np.save(base_folder+'all_qualitative.npy', out)


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    calculate_metrics(args)

