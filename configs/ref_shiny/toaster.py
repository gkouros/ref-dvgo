_base_ = '../default.py'

basedir = './logs/ref_shiny/toaster'
expname = 'output'

data = dict(
    datadir='./data/ref_shiny/toaster',
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
    weight_tv_density=1e-5,
    tv_every=1,
    tv_after=0,
    tv_before=20000,
    tv_dense_before=12000,
)

coarse_model_and_render = dict(
    num_voxels=100**3,
    num_voxels_base=100**3,
)

fine_model_and_render = dict(
    num_voxels=160**3,
    num_voxels_base=160**3,
    rgbnet_dim=16,
    rgbnet_width=128,
    rgbnet_depth=3,
    rgbnet_direct=True,  # change to false for diffuse pretraining
    world_bound_scale=1.05
)
