_base_ = './cascade_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
fp16 = dict(loss_scale=512.)
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)