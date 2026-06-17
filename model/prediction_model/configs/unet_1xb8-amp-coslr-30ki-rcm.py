'''
No@
'''
_base_ = ['unet_1xb8-coslr-30ki_rcm.py']

# mixed precision
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')

wandb_config = _base_.wandb_config
wandb_config.init_kwargs.name = '{{fileBasenameNoExtension}}'
vis_backends = [wandb_config, dict(type='LocalVisBackend')]
visualizer = dict(vis_backends=vis_backends)