_base_ = '../default.py'

basedir = './logs/ref_shiny/teapot'
expname = 'output'

data = dict(
    datadir='./data/ref_shiny/teapot',
    dataset_type='blender',
    white_bkgd=True,
)

coarse_train = dict(
    N_iters=5000,
    pervoxel_lr=True,
)

fine_train = dict(
    N_iters=20000,
    pervoxel_lr=False,
    lrate_density=1e-1,
    lrate_k0=1e-1,
    lrate_grad_pred=1e-1,
    lrate_diffuse=1e-1,
    lrate_tint=1e-1,
    lrate_roughness=1e-1,
    lrate_rgbnet=1e-3,
    lrate_decay=20,
    weight_entropy_last=0.001,
    weight_rgbper=0.01,
    weight_orientation=1e-4,
    weight_pred_normals=3e-8,
    weight_tv_density=1e-5,
    weight_tv_k0=1e-5,
    weight_tv_grad_pred=1e-5,
    weight_tv_tint=1e-5,
    weight_tv_roughness=1e-5,
    weight_tv_diffuse=1e-5,
    tv_every=1,
    tv_after=0,
    tv_before=20000,
    tv_dense_before=12000,
    pg_scale=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
    skip_zero_grad_fields=['density', 'k0', 'diffuse', 'grad_pred', 'tint', 'roughness'],
)

coarse_model_and_render = dict(
    num_voxels=100**3,
    num_voxels_base=100**3,
)

fine_model_and_render = dict(
    num_voxels=160**3,
    num_voxels_base=160**3,
    rgbnet_dim=16,
    rgbnet_width=256,
    rgbnet_depth=8,
    rgbnet_direct=False,
    use_reflections=True,
    use_n_dot_v=True,
    use_specular_tint=True,
    enable_pred_normals=True,
    enable_pred_roughness=True,
    use_sh_encoding=True,
    use_refnerf_architecture=True,
    normals_to_use='normals_pred',
    disable_density_normals=False,
)
