import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange
import json
from copy import deepcopy

import mmcv
import imageio
import numpy as np
from functools import reduce
import operator
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils, dvgo, drefvgo
from lib.load_data import load_data

from torch_efficient_distloss import flatten_eff_distloss


def config_parser():
    '''Define command line arguments
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument('--expname', type=str, default='',
                        help='the experiment id')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')
    parser.add_argument("--export_fine_only", type=str, default='')
    parser.add_argument('--overwrite', type=str, help='Overwrite params in file')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_flipy", action='store_true')
    parser.add_argument("--render_video_rot90", default=0, type=int)
    parser.add_argument("--render_video_factor", type=float, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--dump_images", action='store_true')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    return parser


def overwrite_configs_with_args(config, args):
    if not args.overwrite:
        return
    overwrite = args.overwrite.split()
    overwrite = dict([arg.split('=') for arg in overwrite])
    def getFromDict(dataDict, mapList):
        return reduce(operator.getitem, mapList, dataDict)
    def setInDict(dataDict, mapList, value):
        getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value
    for key in overwrite:
        subkeys = key.split('.')
        try:
            val = eval(overwrite[key])  # numeric
        except NameError as ne:
            print(ne)
            val = overwrite[key].replace('\'', '')  # string
        except TypeError as te:
            print(te)
            val = overwrite[key]  # dict
        setInDict(config, subkeys, val)


@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, dump_images=False,
                      render_factor=0, render_video_flipy=False, render_video_rot90=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs = []
    depths = []
    extras = {}
    bgmaps = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    for i, c2w in enumerate(tqdm(render_poses, desc=f'Rendering views')):

        H, W = HW[i]
        K = Ks[i]
        c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'alphainv_last',
                'diffuse_marched', 'specular_marched', 'tint_marched',
                'normals_marched', 'normals_pred_marched', 'roughness_marched']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)

        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()
        bgmap = render_result['alphainv_last'].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)

        if 'normals_marched' in render_result:
            if not 'normals' in extras:
                extras['normals'] = []
            normals = render_result['normals_marched'].cpu().numpy()
            extras['normals'].append(normals)
        if 'normals_pred_marched' in render_result:
            if not 'normals_pred' in extras:
                extras['normals_pred'] = []
            normals_pred = render_result['normals_pred_marched'].cpu().numpy()
            extras['normals_pred'].append(normals_pred)
        if 'diffuse_marched' in render_result:
            if not 'diffuse' in extras:
                extras['diffuse'] = []
            diffuse = render_result['diffuse_marched'].cpu().numpy()
            extras['diffuse'].append(diffuse)
        if 'specular_marched' in render_result:
            if not 'specular' in extras:
                extras['specular'] = []
            specular = render_result['specular_marched'].cpu().numpy()
            extras['specular'].append(specular)
        if 'tint_marched' in render_result:
            tint = render_result['tint_marched'].cpu().numpy()
            if not 'tint' in extras:
                extras['tint'] = []
            extras['tint'].append(tint)
        if 'roughness_marched' in render_result:
            if not 'roughness' in extras:
                extras['roughness'] = []
            roughness = render_result['roughness_marched'].cpu().numpy()
            extras['roughness'].append(roughness)

        if i==0:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

    if len(psnrs):
        results = {'psnr': np.mean(psnrs), 'ssim': np.mean(ssims)}
        if eval_lpips_vgg: results['lpips (vgg)'] = np.mean(lpips_vgg)
        if eval_lpips_alex: results['lpips (alex)'] = np.mean(lpips_alex)
        print('Evaluation Metrics: ', results)
        import json
        with open(os.path.join(savedir, '../evaluation.txt'), 'w') as fp:
            json.dump(results, fp)

    if render_video_flipy:
        for i in range(len(rgbs)):
            rgbs[i] = np.flip(rgbs[i], axis=0)
            depths[i] = np.flip(depths[i], axis=0)
            bgmaps[i] = np.flip(bgmaps[i], axis=0)
            for key in extras:
                extras[key][i] = np.flip(extras[key][i], axis=0)

    if render_video_rot90 != 0:
        for i in range(len(rgbs)):
            rgbs[i] = np.rot90(rgbs[i], k=render_video_rot90, axes=(0,1))
            depths[i] = np.rot90(depths[i], k=render_video_rot90, axes=(0,1))
            bgmaps[i] = np.rot90(bgmaps[i], k=render_video_rot90, axes=(0,1))
            for key in extras:
                extras[key][i] = np.rot90(extras[key][i], k=render_video_rot90, axes=(0,1))

    if savedir is not None and dump_images:
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, 'color_{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

            depth8 = utils.to8b(depths[i])
            depth8 = np.concatenate((depth8,) * 3, axis=-1)
            filename = os.path.join(savedir, 'disp_{:03d}.png'.format(i))
            imageio.imwrite(filename, depth8)

            depth = np.concatenate((depths[i],) * 3, axis=-1)
            filename = os.path.join(savedir, 'disp_{:03d}.tiff'.format(i))
            imageio.imwrite(filename, depth)

            for key in extras:
                if key.startswith('normals'):
                    extras[key][i] = matte(extras[key][i] / 2. + 0.5, bgmaps[i])
                else:
                    extras[key][i] = matte(extras[key][i], bgmaps[i])

                extra8 = utils.to8b(extras[key][i])
                if extra8.shape[-1] == 1:
                    extra8 = np.concatenate((extra8,) * 3, axis=-1)
                filename = os.path.join(savedir, f'{key}_{i:03d}.png')
                imageio.imwrite(filename, extra8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)
    extras = {key: np.array(extras[key]) for key in extras}

    return rgbs, depths, bgmaps, extras


def matte(vis, bgmap, dark=0.8, light=1.0, width=8):
    """Set non-accumulated pixels to a Photoshop-esque checker pattern."""
    acc = 1.0 - bgmap
    bg_mask = np.logical_xor(
        (np.arange(acc.shape[0]) % (2 * width) // width)[:, None],
        (np.arange(acc.shape[1]) % (2 * width) // width)[None, :])
    bg = np.where(~bg_mask, light, dark)[..., None]
    return vis * acc + (bg * (1 - acc))


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)

    # remove useless field
    kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far', 'near_clip',
            'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images', 'diffuse'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
        if cfg.data.load_diffuse:
            data_dict['diffuse'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['diffuse']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
        if cfg.data.load_diffuse:
            data_dict['diffuse'] = torch.FloatTensor(data_dict['diffuse'], device='cpu')

    data_dict['poses'] = torch.Tensor(data_dict['poses'])

    return data_dict


def _compute_bbox_by_cam_frustrm_bounded(cfg, HW, Ks, poses, i_train, near, far):
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        if cfg.data.ndc:
            pts_nf = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        else:
            pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    return xyz_min, xyz_max


def _compute_bbox_by_cam_frustrm_unbounded(cfg, HW, Ks, poses, i_train, near_clip):
    # Find a tightest cube that cover all camera centers
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        pts = rays_o + rays_d * near_clip
        xyz_min = torch.minimum(xyz_min, pts.amin((0,1)))
        xyz_max = torch.maximum(xyz_max, pts.amax((0,1)))
    center = (xyz_min + xyz_max) * 0.5
    radius = (center - xyz_min).max() * cfg.data.unbounded_inner_r
    xyz_min = center - radius
    xyz_max = center + radius
    return xyz_min, xyz_max


def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    if cfg.data.unbounded_inward:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_unbounded(
                cfg, HW, Ks, poses, i_train, kwargs.get('near_clip', None))
    else:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_bounded(
                cfg, HW, Ks, poses, i_train, near, far)
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max


@torch.no_grad()
def compute_bbox_by_coarse_geo(model_class, model_path, thres):
    print('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()
    model = utils.load_model(model_class, model_path)
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.world_size[0]),
        torch.linspace(0, 1, model.world_size[1]),
        torch.linspace(0, 1, model.world_size[2]),
    ), -1)
    dense_xyz = model.xyz_min * (1-interp) + model.xyz_max * interp
    density = model.density(dense_xyz)
    alpha = model.activate_density(density)
    mask = (alpha > thres)
    active_xyz = dense_xyz[mask]
    print(active_xyz)
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
    print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)
    eps_time = time.time() - eps_time
    print('compute_bbox_by_coarse_geo: finish (eps time:', eps_time, 'secs)')
    return xyz_min, xyz_max


def create_new_model(cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage,
                     device, coarse_ckpt_path):
    model_kwargs = copy.deepcopy(cfg_model)
    num_voxels = model_kwargs.pop('num_voxels')
    if len(cfg_train.pg_scale):
        num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))

    if cfg_model.use_reflections:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse ref voxel grid (covering unbounded)\033[0m')
        model = drefvgo.DirectRefVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            mask_cache_path=coarse_ckpt_path,
            **model_kwargs)
    else:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse dense voxel grid\033[0m')
        model = dvgo.DirectVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            mask_cache_path=coarse_ckpt_path,
            **model_kwargs)

    model = model.to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    return model, optimizer


def load_existing_model(stage, args, cfg, cfg_train, cfg_model, reload_ckpt_path, device):
    if cfg_model.use_reflections:
        model_class = drefvgo.DirectRefVoxGO
    else:
        model_class = dvgo.DirectVoxGO
    model = utils.load_model(model_class, reload_ckpt_path).to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    model, optimizer, start = utils.load_checkpoint(
            model, optimizer, reload_ckpt_path, args.no_reload_optimizer)
    return model, optimizer, start


def orientation_loss(render_result, cfg_model):
    """Computes the orientation loss regularizer defined in ref-NeRF."""
    zero = torch.tensor(0.0, dtype=torch.float32)
    w = render_result['weights'].detach()
    n = render_result[cfg_model.normals_to_use]
    # Negate viewdirs to represent normalized vectors from point to camera.
    v = -render_result['viewdirs']
    n_dot_v = (n * v).sum(dim=-1)
    zeros = torch.zeros_like(n_dot_v)
    return torch.mean((w * torch.fmin(zero, n_dot_v)**2).sum(dim=-1))


def predicted_normal_loss(render_result):
    """Computes the predicted normal supervision loss defined in ref-NeRF."""
    w = render_result['weights'].detach()
    n = render_result['normals'].detach()
    npred = render_result['normals_pred']
    return torch.mean((w * (1.0 - torch.sum(n * npred, dim=-1))).sum(dim=-1))


def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage,
                             coarse_ckpt_path=None):
    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        print(f'xyz_min: {xyz_min}, xyz_max: {xyz_min}')
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift

    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images'
        ]
    ]
    if 'depth' in data_dict:
        depth_images = torch.tensor(np.array(data_dict['depth']))
        depth_tr_ori = depth_images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)
    else:
        depth_tr_ori = None

    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
    if args.no_reload:
        reload_ckpt_path = None
    elif args.ft_path:
        reload_ckpt_path = args.ft_path
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None

    # init model and optimizer
    if reload_ckpt_path is None:
        print(f'scene_rep_reconstruction ({stage}): train from scratch')
        model, optimizer = create_new_model(
            cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, device,
            coarse_ckpt_path)
        start = 0
        if cfg_model.maskout_near_cam_vox:
            model.maskout_near_cam_vox(poses[i_train,:3,3], near)
    else:
        print(f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        model, optimizer, start = load_existing_model(
            stage, args, cfg, cfg_train, cfg_model, reload_ckpt_path, device)

    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'rand_bkgd': cfg.data.rand_bkgd,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
        'render_depth': cfg_train.weight_depth > 0,
        'no_grad_depth': cfg_train.weight_depth > 0,
        'render_features': False,
    }

    # init batch rays sampler
    def gather_training_rays():
        if data_dict['irregular_shape']:
            rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
        else:
            rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

        if cfg_train.ray_sampler == 'in_maskcache':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_in_maskcache_sampling(
                    rgb_tr_ori=rgb_tr_ori,
                    train_poses=poses[i_train],
                    HW=HW[i_train], Ks=Ks[i_train],
                    ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                    model=model, render_kwargs=render_kwargs)
        elif cfg_train.ray_sampler == 'flatten':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        else:
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays(
                rgb_tr=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        index_generator = dvgo.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler

    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()

    # view-count-based learning rate
    if cfg_train.pervoxel_lr:
        def per_voxel_init():
            cnt = model.voxel_count_views(
                    rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=near, far=far,
                    stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                    irregular_shape=data_dict['irregular_shape'])
            optimizer.set_pervoxel_lr(cnt)
            model.mask_cache.mask[cnt.squeeze() <= 2] = False
        per_voxel_init()

    if cfg_train.maskout_lt_nviews > 0:
        model.update_occupancy_cache_lt_nviews(
                rays_o_tr, rays_d_tr, imsz, render_kwargs, cfg_train.maskout_lt_nviews)

    # GOGO
    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1
    loop = trange(1+start, 1+cfg_train.N_iters, desc=f'({stage})')
    for global_step in loop:

        # renew occupancy grid
        if model.mask_cache is not None and (global_step + 500) % 1000 == 0:
            model.update_occupancy_cache()

        # progress scaling checkpoint
        if global_step in cfg_train.pg_scale:
            n_rest_scales = len(cfg_train.pg_scale)-cfg_train.pg_scale.index(global_step)-1
            cur_voxels = int(cfg_model.num_voxels / (2**n_rest_scales))
            model.scale_volume_grid(cur_voxels)
            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
            model.act_shift -= cfg_train.decay_after_scale
            torch.cuda.empty_cache()

        # random sample rays
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache']:
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]
        elif cfg_train.ray_sampler == 'random':
            sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
        else:
            raise NotImplementedError

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)

        # volume rendering
        render_result = model(
            rays_o, rays_d, viewdirs,
            global_step=global_step, is_train=True,
            **render_kwargs)

        # gradient descent step
        optimizer.zero_grad(set_to_none=True)
        loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)
        psnr = utils.mse2psnr(loss.detach())
        losses = {'phot': loss.detach()}
        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            bkgd_loss = cfg_train.weight_entropy_last * entropy_last_loss
            loss += bkgd_loss
            losses['bkgd'] = bkgd_loss.detach()
        if cfg_train.weight_nearclip > 0:
            near_thres = data_dict['near_clip'] / model.scene_radius[0].item()
            near_mask = (render_result['t'] < near_thres)
            density = render_result['raw_density'][near_mask]
            if len(density):
                nearclip_loss = cfg_train.weight_nearclip * (density - density.detach()).sum()
                loss += nearclip_loss
                losses['nearclip'] = nearclip_loss.detach()
        if cfg_train.weight_distortion > 0:
            s = render_result['s']
            w = render_result['weights']
            ray_id = render_result['ray_id']
            if 'interval' in render_result:
                interval = render_result['interval']
            elif 'n_max' in render_result:
                n_max = render_result['n_max']
                interval = 1 / n_max
            else:
                raise KeyError('No interval information in rendered result')
            distortion_loss = cfg_train.weight_distortion * flatten_eff_distloss(w, s, interval, ray_id)
            loss += distortion_loss
            losses['dist'] = distortion_loss.detach()
        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
            rgbper_loss = cfg_train.weight_rgbper * (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
            loss += rgbper_loss
            losses['rgbper'] = rgbper_loss.detach()
        if cfg_model.enable_pred_normals and cfg_train.weight_orientation > 0:
            ori_loss = cfg_train.weight_orientation * orientation_loss(render_result, cfg_model)
            loss += ori_loss
            losses['ori'] = ori_loss.detach()
        if cfg_model.enable_pred_normals and cfg_train.weight_pred_normals > 0:
            npred_loss = cfg_train.weight_pred_normals * predicted_normal_loss(render_result)
            loss += npred_loss
            losses['npred'] = npred_loss.detach()

        # backpropagate total loss
        loss.backward()

        if global_step<cfg_train.tv_before and global_step>cfg_train.tv_after and global_step%cfg_train.tv_every==0:
            if cfg_train.weight_tv_density > 0:
                model.density_total_variation_add_grad(
                    cfg_train.weight_tv_density/len(rays_o), global_step<cfg_train.tv_dense_before)
            if cfg_train.weight_tv_k0 > 0 and hasattr(model, 'k0'):
                model.k0_total_variation_add_grad(
                    cfg_train.weight_tv_k0/len(rays_o), global_step<cfg_train.tv_dense_before)
            if cfg_train.weight_tv_grad_pred > 0 and hasattr(model, 'grad_pred'):
                model.grad_pred_total_variation_add_grad(
                    cfg_train.weight_tv_grad_pred/len(rays_o), global_step<cfg_train.tv_dense_before)
            if cfg_train.weight_tv_tint > 0 and hasattr(model, 'tint'):
                model.tint_total_variation_add_grad(
                    cfg_train.weight_tv_tint/len(rays_o), global_step<cfg_train.tv_dense_before)
            if cfg_train.weight_tv_roughness > 0 and hasattr(model, 'roughness'):
                model.roughness_total_variation_add_grad(
                    cfg_train.weight_tv_roughness/len(rays_o), global_step<cfg_train.tv_dense_before)
            if cfg_train.weight_tv_diffuse > 0 and hasattr(model, 'diffuse'):
                model.diffuse_total_variation_add_grad(
                    cfg_train.weight_tv_diffuse/len(rays_o), global_step<cfg_train.tv_dense_before)

        optimizer.step()
        psnr_lst.append(psnr.item())

        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor

        # check log & save
        if global_step%args.i_print==0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            # loop.set_description(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
            #       f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
            #       f'Eps: {eps_time_str}')
            printed_losses = {k: losses[k].item() for k in losses}
            print(f'Total Loss: {loss.item():.5f}, PSNR: {np.mean(psnr_lst):3.2f}, Losses={printed_losses}')
            psnr_lst = []

        if global_step%args.i_weights==0:
            path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_{global_step:06d}.tar')
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', path)

    if global_step != -1:
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)


def train(args, cfg, data_dict):
    """ main training routine """

    # init
    print('train: start')
    eps_time = time.time()
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    # coarse geometry searching (only works for inward bounded scenes)
    eps_coarse = time.time()
    xyz_min_coarse, xyz_max_coarse = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
    if cfg.coarse_train.N_iters > 0:
        scene_rep_reconstruction(
                args=args, cfg=cfg,
                cfg_model=cfg.coarse_model_and_render, cfg_train=cfg.coarse_train,
                xyz_min=xyz_min_coarse, xyz_max=xyz_max_coarse,
                data_dict=data_dict, stage='coarse')
        eps_coarse = time.time() - eps_coarse
        eps_time_str = f'{eps_coarse//3600:02.0f}:{eps_coarse//60%60:02.0f}:{eps_coarse%60:02.0f}'
        print('train: coarse geometry searching in', eps_time_str)
        coarse_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'coarse_last.tar')
    else:
        print('train: skip coarse geometry searching')
        coarse_ckpt_path = None

    eps_fine = time.time()
    if cfg.coarse_train.N_iters == 0 or cfg.data.ndc:
        xyz_min_fine, xyz_max_fine = xyz_min_coarse.clone(), xyz_max_coarse.clone()
    else:
        xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo(
                model_class=dvgo.DirectVoxGO, model_path=coarse_ckpt_path,
                thres=cfg.fine_model_and_render.bbox_thres)

    # fine detail reconstruction
    scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
            xyz_min=xyz_min_fine.clone(), xyz_max=xyz_max_fine.clone(),  # clone to avoid modification inside function
            data_dict=data_dict, stage='fine',
            coarse_ckpt_path=coarse_ckpt_path)
    eps_fine = time.time() - eps_fine
    eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
    print('train: fine detail reconstruction in', eps_time_str)

    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    print('train: finish (eps time', eps_time_str, ')')


def main(args, cfg):
    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()

    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg)

    # export scene bbox and camera poses in 3d for debugging and visualization
    if args.export_bbox_and_cams_only:
        print('Export bbox and cameras...')
        xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
        poses, HW, Ks, i_train = data_dict['poses'], data_dict['HW'], data_dict['Ks'], data_dict['i_train']
        near, far = data_dict['near'], data_dict['far']
        if data_dict['near_clip'] is not None:
            near = data_dict['near_clip']
        cam_lst = []
        for c2w, (H, W), K in zip(poses[i_train], HW[i_train], Ks[i_train]):
            rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                    H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
            cam_o = rays_o[0,0].cpu().numpy()
            cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
            cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))
        ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_last.tar')
        xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo(
            model_class=dvgo.DirectVoxGO, model_path=ckpt_path,
            thres=cfg.fine_model_and_render.bbox_thres)
        export_path = os.path.join(cfg.basedir, cfg.expname, args.export_bbox_and_cams_only)
        np.savez_compressed(export_path,
            xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
            xyz_min_fine=xyz_min_fine.cpu(), xyz_max_fine=xyz_max_fine.cpu(),
            cam_lst=np.array(cam_lst))
        print('done')
        sys.exit()

    if args.export_coarse_only:
        print('Export coarse visualization...')
        with torch.no_grad():
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_last.tar')
            model = utils.load_model(dvgo.DirectVoxGO, ckpt_path).to(device)
            alpha = model.activate_density(model.density.get_dense_grid()).squeeze().cpu().numpy()
            rgb = torch.sigmoid(model.k0.get_dense_grid()).squeeze().permute(1,2,3,0).cpu().numpy()

        export_path = os.path.join(cfg.basedir, cfg.expname, args.export_coarse_only)
        np.savez_compressed(export_path, alpha=alpha, rgb=rgb)
        print('done')
        sys.exit()

    if args.export_fine_only:
        print('Export coarse visualization...')
        with torch.no_grad():
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
            model = utils.load_model(dvgo.DirectVoxGO, ckpt_path).to(device)
            alpha = model.activate_density(model.density.get_dense_grid()).squeeze().cpu().numpy()
            rgb = torch.sigmoid(model.k0.get_dense_grid()).squeeze().permute(1,2,3,0).cpu().numpy()

        export_path = os.path.join(cfg.basedir, cfg.expname, args.export_fine_only)
        np.savez_compressed(export_path, alpha=alpha, rgb=rgb)
        print('done')
        sys.exit()

    # train
    if not args.render_only:
        train(args, cfg, data_dict)

    # load model for rendering
    if args.render_test or args.render_train or args.render_video:
        if args.ft_path:
            ckpt_path = args.ft_path
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        if cfg.fine_model_and_render.use_reflections:
            model_class = drefvgo.DirectRefVoxGO
        else:
            model_class = dvgo.DirectVoxGO

        model = utils.load_model(model_class, ckpt_path).to(device)
        model.eval()
        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_depth': True,
                'render_diffuse': True,
                'render_tint': True,
                'render_roughness': True,
                'render_normals': False,
                'render_normals_pred': True,
                'render_specular': True,
            },
        }

    # specify whether to render and evaluate on diffuse or regular images
    images_key = 'diffuse' if cfg.fine_train.diffuse_only else 'images'

    # render trainset and eval
    if args.render_train or cfg.fine_train.diffuse_only:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps, extras = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_train']],
                HW=data_dict['HW'][data_dict['i_train']],
                Ks=data_dict['Ks'][data_dict['i_train']],
                gt_imgs=[data_dict[images_key][i].cpu().numpy() for i in data_dict['i_train']],
                savedir=testsavedir, dump_images=args.dump_images or cfg.fine_train.diffuse_only,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)
        import matplotlib.pyplot as plt
        depths_vis = depths * (1-bgmaps) + bgmaps
        dmin, dmax = np.percentile(depths_vis[bgmaps < 0.1], q=[5, 95])
        depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depths_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth_rainbow.mp4'), utils.to8b(depth_vis), fps=30, quality=8)
        if 'normals' in extras:
            imageio.mimwrite(os.path.join(testsavedir, 'video.normals.mp4'), utils.to8b(extras['normals']), fps=30, quality=8)
        if 'normals_pred' in extras:
            imageio.mimwrite(os.path.join(testsavedir, 'video.normals_pred.mp4'), utils.to8b(extras['normals_pred']), fps=30, quality=8)
        if 'diffuse' in extras:
            imageio.mimwrite(os.path.join(testsavedir, 'video.diffuse.mp4'), utils.to8b(extras['diffuse']), fps=30, quality=8)
        if 'specular' in extras:
            imageio.mimwrite(os.path.join(testsavedir, 'video.specular.mp4'), utils.to8b(extras['specular']), fps=30, quality=8)
        if 'tint' in extras:
            imageio.mimwrite(os.path.join(testsavedir, 'video.tint.mp4'), utils.to8b(extras['tint']), fps=30, quality=8)
        if 'roughness' in extras:
            imageio.mimwrite(os.path.join(testsavedir, 'video.roughness.mp4'), utils.to8b(extras['roughness']), fps=30, quality=8)


    # render testset and eval
    if args.render_test:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps, extras = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_test']],
                HW=data_dict['HW'][data_dict['i_test']],
                Ks=data_dict['Ks'][data_dict['i_test']],
                gt_imgs=[data_dict[images_key][i].cpu().numpy() for i in data_dict['i_test']],
                savedir=testsavedir, dump_images=args.dump_images,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)
        import matplotlib.pyplot as plt
        depths_vis = depths * (1-bgmaps) + bgmaps
        dmin, dmax = np.percentile(depths_vis[bgmaps < 0.1], q=[5, 95])
        depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depths_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth_rainbow.mp4'), utils.to8b(depth_vis), fps=30, quality=8)
        if 'normals' in extras:
            imageio.mimwrite(os.path.join(testsavedir, 'video.normals.mp4'), utils.to8b(extras['normals']), fps=30, quality=8)
        if 'normals_pred' in extras:
            imageio.mimwrite(os.path.join(testsavedir, 'video.normals_pred.mp4'), utils.to8b(extras['normals_pred']), fps=30, quality=8)
        if 'diffuse' in extras:
            imageio.mimwrite(os.path.join(testsavedir, 'video.diffuse.mp4'), utils.to8b(extras['diffuse']), fps=30, quality=8)
        if 'specular' in extras:
            imageio.mimwrite(os.path.join(testsavedir, 'video.specular.mp4'), utils.to8b(extras['specular']), fps=30, quality=8)
        if 'tint' in extras:
            imageio.mimwrite(os.path.join(testsavedir, 'video.tint.mp4'), utils.to8b(extras['tint']), fps=30, quality=8)
        if 'roughness' in extras:
            imageio.mimwrite(os.path.join(testsavedir, 'video.roughness.mp4'), utils.to8b(extras['roughness']), fps=30, quality=8)

    # render video
    if args.render_video:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps, extras = render_viewpoints(
                render_poses=data_dict['render_poses'],
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                render_factor=args.render_video_factor,
                render_video_flipy=args.render_video_flipy,
                render_video_rot90=args.render_video_rot90,
                savedir=testsavedir, dump_images=args.dump_images,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        import matplotlib.pyplot as plt
        depths_vis = depths * (1-bgmaps) + bgmaps
        dmin, dmax = np.percentile(depths_vis[bgmaps < 0.1], q=[5, 95])
        depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depths_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(depth_vis), fps=30, quality=8)
        if 'normals' in extras:
            imageio.mimwrite(os.path.join(testsavedir, 'video.normals.mp4'), utils.to8b(extras['normals']), fps=30, quality=8)
        if 'normals_pred' in extras:
            imageio.mimwrite(os.path.join(testsavedir, 'video.normals_pred.mp4'), utils.to8b(extras['normals_pred']), fps=30, quality=8)
        if 'diffuse' in extras:
            imageio.mimwrite(os.path.join(testsavedir, 'video.diffuse.mp4'), utils.to8b(extras['diffuse']), fps=30, quality=8)
        if 'specular' in extras:
            imageio.mimwrite(os.path.join(testsavedir, 'video.specular.mp4'), utils.to8b(extras['specular']), fps=30, quality=8)
        if 'tint' in extras:
            imageio.mimwrite(os.path.join(testsavedir, 'video.tint.mp4'), utils.to8b(extras['tint']), fps=30, quality=8)
        if 'roughness' in extras:
            imageio.mimwrite(os.path.join(testsavedir, 'video.roughness.mp4'), utils.to8b(extras['roughness']), fps=30, quality=8)

    print('Done')


if __name__=='__main__':
    # load setup
    parser = config_parser()
    args = parser.parse_args()
    try:
        cfg = mmcv.Config.fromfile(args.config)
    except AttributeError:
        import mmengine
        cfg = mmengine.Config.fromfile(args.config)
    if 'overwrite' in args:
        overwrite_configs_with_args(cfg, args)
    if args.expname != '':
        cfg.expname = args.expname

    main(args, cfg)
