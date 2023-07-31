import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1, radius=4.0, load_diffuse=False):
    splits = ['train', 'val', 'test']
    metas = {}
    blacklists = {}
    for s in splits:
        try:
            with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
                metas[s] = json.load(fp)

            blacklist_path = os.path.join(basedir, 'blacklist_{}.txt'.format(s))
            if os.path.exists(blacklist_path):
                with open(blacklist_path, 'r') as fp:
                    blacklists[s] = fp.read().split('\n')[:-1]

                metas[s]['frames'] = [
                    x for x in metas[s]['frames']
                    if x['file_path'].split('/')[1] not in blacklists[s]]
        except FileNotFoundError:
            if s == 'val':
                with open(os.path.join(basedir, 'transforms_test.json'.format(s)), 'r') as fp:
                    metas['val'] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_diffs = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        diffs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
            if load_diffuse:
                dfname = fname.replace('.png', '_diffuse.png')
                diffs.append(imageio.imread(dfname))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        all_imgs.append(imgs)
        poses = np.array(poses).astype(np.float32)
        all_poses.append(poses)
        if load_diffuse:
            diffs = (np.array(diffs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
            all_diffs.append(diffs)

        counts.append(counts[-1] + imgs.shape[0])

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    diffs = np.concatenate(all_diffs, 0) if load_diffuse else []

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, radius) for angle in np.linspace(-180,180,160+1)[:-1]], 0)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    return imgs, diffs, poses, render_poses, [H, W, focal], i_split
