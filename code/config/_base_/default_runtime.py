checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
#load_from = None
load_from = r'D:\github\mmdetection_v2.1.0\mmdetection-2.1.0\checkpoints\cascade_rcnn_r50_dcn_coco_pretrained_weights_classes_21.pth'  # 采用coco预训练模型 ,需要对权重类别数进行处理
resume_from = None
workflow = [('train', 1)]
