from mmdet.apis import init_detector
from mmdet.apis import inference_detector
from mmdet.apis import show_result_pyplot
import time
# 模型配置文件
config_file = '../../configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
#config_file = '../../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

# 预训练模型文件
checkpoint_file = '../../checkpoints/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'
#checkpoint_file = '../../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# 通过模型配置文件与预训练文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')
# print(model)
# 测试单张图片
start = time.time()
img = '../../demo/demo.jpg'
result = inference_detector(model, img)
# show_result_pyplot(img, result, model.CLASSES)
end = time.time()
t = end-start
print(t)
show_result_pyplot(model, img, result)

