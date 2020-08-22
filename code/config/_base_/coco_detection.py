dataset_type = 'CocoDataset'
data_root = 'D:/dataset/data/skypool_first_trainval/'    #训练数据路径
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='Resize', img_scale=(2446, 1000), keep_ratio=True),
    #多尺度训练
    dict(type='Resize',
         multiscale_mode="value",
         img_scale=[(2446, 1000), (1600,1000),(1200,100)],
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomVFlip', flip_ratio=0.5),
    dict(type='BBoxJitter', min=0.9, max=1.1),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        #img_scale=(2446, 1000),
        #多尺度测试
        img_scale=[(2446, 1000),(1600,1000),(1200,1000)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='RandomVFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,     # 每张gpu训练多少张图片  batch_size = gpu_num(训练使用gpu数量) * imgs_per_gpu
    workers_per_gpu=1,     #有多少个gpu
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
