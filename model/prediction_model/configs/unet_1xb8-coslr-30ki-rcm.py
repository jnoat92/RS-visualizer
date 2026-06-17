'''
No@
'''
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

# ============== DATASET ==============
import os
import numpy as np

crop_size = (384, 384)
pixel_spacing_train = [200]   # List all pixel spacings (in meters) to include during training
pixel_spacing_test = 200

GT_type = ['pixel_labels']

# dataset settings
dataset_type_train = 'RCMPatches'
dataset_type_val = 'RCMPatches'

# data_root_train_nc = '/home/jnoat92/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3'
# gt_root_train = '/home/jnoat92/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3_segmaps'
# data_root_test_nc = '/home/jnoat92/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_test_v3'
# gt_root_test = '/home/jnoat92/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_test_v3_segmaps'
data_root_patches = 'D:/Temp/Model_update_code/RCM_dataset/patches'

file_train = 'D:/Temp/Model_update_code/RCM_dataset/data_info/train_80.txt'
file_val   = 'D:/Temp/Model_update_code/RCM_dataset/data_info/val_20.txt'
# file_test = 'D:/Temp/Model_update_code/RCM_dataset\data_info/test_file.txt'


# channels to use
channels = [
    # RCM SAR channels #
    'HH',
    'HV',
]


# ------------- TRAIN SETUP
train_pipeline = [
    dict(type='LoadPatchFromPKLFile', channels=channels, 
         to_float32=True, nan=255, with_seg=True, GT_type=GT_type, 
         mean=[-16.849971303873264, -27.80859960890284]),
    # dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=crop_size,
        ratio_range=(1.0, 1.1),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.9),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion')
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                                          'img_shape', 'pad_shape', 'scale_factor', 'flip',
                                          'flip_direction', 'reduce_zero_label', 
                                          'segments', 'n_segments', 'boundaries')) 
                                            # 'segments', 'n_segments', and 'boundaries' are the only non-default 
                                            # parameter. W need it when using the ITT framework
]

train_concat_dataset = dict(type='ConcatDataset', 
                      datasets= [dict(type=dataset_type_train,
                                      data_root = os.path.join(data_root_patches, 'pixel_spacing_%dm'%(i)),
                                      ann_file = file_train,
                                      pipeline = train_pipeline) for i in pixel_spacing_train])
train_dataloader = dict(batch_size=8,
                        num_workers=8,
                        persistent_workers=True,
                        sampler=dict(type='WeightedInfiniteSampler', use_weights=True),
                        # sampler=dict(type='InfiniteSampler', shuffle=True),
                        dataset=train_concat_dataset)



# ------------- VAL SETUP
val_pipeline = [
    dict(type='LoadPatchFromPKLFile', channels=channels, 
         to_float32=True, nan=255, with_seg=True, GT_type=GT_type, 
         mean=[-16.849971303873264, -27.80859960890284]),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                                          'img_shape', 'pad_shape', 'scale_factor', 'flip',
                                          'flip_direction', 'reduce_zero_label', 
                                          'segments', 'n_segments', 'boundaries')) 
                                            # 'segments', 'n_segments', and 'boundaries' are the only non-default 
                                            # parameter. W need it when using the ITT framework
]

val_concat_dataset = dict(type='ConcatDataset', 
                      datasets= [dict(type=dataset_type_train,
                                      data_root = os.path.join(data_root_patches, 'pixel_spacing_%dm'%(i)),
                                      ann_file = file_val,
                                      pipeline = val_pipeline) for i in pixel_spacing_train])
val_dataloader = dict(batch_size=8,
                        num_workers=8,
                        persistent_workers=True,
                        sampler=dict(type='DefaultSampler', shuffle=False),
                        dataset=val_concat_dataset)


# # ------------- TEST SETUP
# test_pipeline = val_pipeline
# test_pipeline = [
#     dict(type='PreLoadImageandSegFromNetCDFFile', data_root=data_root_test_nc, gt_root=gt_root_test, 
#          ann_file=file_test, channels=channels, mean=mean, std=std, to_float32=True, nan=255, 
#          downsample_factor=downsample_factor_test, downsample_factor_for_metrics=5, with_seg=True, GT_type=GT_type),
#     dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape',
#                                           'img_shape', 'pad_shape', 'scale_factor', 'flip',
#                                           'flip_direction', 'reduce_zero_label', 'dws_factor',
#                                           'dws_factor_for_metrics')) 
#                                             # 'dws_factor' and dws_factor_for_metrics are the only non-default 
#                                             # parameter. W need it in the visualization and metric hooks
# ]
# test_dataloader = dict(batch_size=1,
#                       num_workers=1,
#                       persistent_workers=True,
#                       sampler=dict(type='DefaultSampler', shuffle=False),
#                       dataset=dict(type=dataset_type_val,
#                                    data_root=data_root_test_nc,
#                                    ann_file=file_test,
#                                    pipeline=test_pipeline))
test_cfg = None

# ============== MODEL ==============
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[-16.849971303873264, -27.80859960890284],
    std=[6.601613867039019, 6.390188051517371],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=16)) # test_cfg into data_preprocessor provides 
                                    # automatic padding required for predictions in mode 'whole'

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    # pretrained='/project/6075102/AI4arctic/m32patel/mmselfsup/work_dirs/selfsup/mae_vit-base-p16/epoch_200.pth',
    backbone=dict(
        type='UNet',
        base_channels=32,
        upsample_cfg=dict(type='DeconvModule'),
        in_channels=len(channels),
        norm_cfg=norm_cfg),
    neck=None,
    decode_head=#[
        dict(
            type='FCNHead',
            # task='pixel_labels',
            num_classes=2,
            num_convs=0,
            concat_input=False,
            in_channels=32,
            in_index=-1,
            channels=32,
            dropout_ratio=0.1,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1, avg_non_ignore=True)
            ),
        #],
    auxiliary_head = None,
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')  # yapf: disable
    # test_cfg=dict(mode='slide', crop_size=crop_size, stride=(crop_size[0] *5//100, crop_size[1]*5//100))
    )



val_evaluator = dict(type='IoUMetric', iou_metrics=['mFscore', 'mIoU'],)
# test_evaluator = val_evaluator

# ============== SCHEDULE ==============
# AdamW optimizer
# Using 4 GPUs
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))

# runtime settings
n_iterations = 10000
val_interval = 500
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=n_iterations, val_interval=val_interval)

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1/3,
        by_epoch=False,
        begin=0,
        end=n_iterations * 5//100
        ),
    dict(
        type='CosineAnnealingLR',
        by_epoch=False,
        begin=n_iterations * 5//100,
        end=n_iterations
        )
]

# ============== RUNTIME ==============
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='CustomLoggerHook', 
                interval = val_interval//10 if val_interval//10> 0 else 1, 
                log_metric_by_epoch=False),
    runtime_info=dict(type='RuntimeInfoHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CustomCheckpointHook', 
                    save_best=['mFscore'], 
                    rule="greater",
                    by_epoch=False, 
                    interval=-1, save_last=True,
                    max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # early_stopping=dict(type='EarlyStoppingHookMain', 
    #                 monitor="Fscore", rule="greater",
    #                 min_delta=0.0, patience=15),
    # visualization=dict(type='SegAI4ArcticVisualizationHook', 
    #                    tasks=GT_type, num_classes=num_classes, 
    #                    downsample_factor=None, metrics=metrics, 
    #                    combined_score_weights=combined_score_weights, 
    #                    draw=True),
    )

log_processor = dict(type='LogProcessor', log_with_hierarchy=True)  # log_with_hierarchy allows separating metrics 
                                                                    # (train-val-test) in loggers like Tensorboard or Wandb
wandb_config = dict(type='WandbVisBackend',
                     init_kwargs=dict(
                         entity='jnoat92',
                         project='ArcticScope',
                         group="",
                         name='{{fileBasenameNoExtension}}',),
                     define_metric_cfg={'val/Fscore': 'max'},
                     commit=True,
                     log_code_name=None,
                     watch_kwargs=None)
vis_backends = [wandb_config, dict(type='LocalVisBackend')]
visualizer = dict(vis_backends=vis_backends)


custom_imports = dict(
    imports=[
            'mmseg.datasets.rcm_patches',
            'mmseg.datasets.transforms.loading_rcm_patches',
            'mmseg.structures.sampler.multires_sampler',
            
            # 'mmseg.models.segmentors.mutitask_encoder_decoder',
            'mmseg.engine.hooks.custom_logger_hook',
            'mmseg.engine.hooks.custom_checkpoint_hook',

            # 'mmseg.models.backbones.custom_vit_bckbn',
            # 'mmseg.models.losses.mse_loss',
            # 'mmseg.engine.hooks.ai4arctic_runtime_hook',
            # 'mmseg.evaluation.metrics.multitask_ai4arctic_metric',
            # 'mmseg.engine.hooks.ai4arctic_visualization_hook',
            # 'mmseg.engine.hooks.early_stopping_hook_main',
            ],
    allow_failed_imports=False)

# randomness
randomness = dict(seed=1, diff_rank_seed=True)