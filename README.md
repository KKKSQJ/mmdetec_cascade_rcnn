# mmdetec_cascade_rcnn
该代码算法：cascade_rcnn_r50fpn_dconv_c3-c5_1x_coco。baseline:mmdetectionV2.0+

## 安装
### Windows版本安装
参考网址：https://zhuanlan.zhihu.com/p/159912450

### 环境说明
- windwos
- vs: 2015 or 2017
- mmdetection: 2.1.0
- mmcv: 0.6.0
- [torch=1.3.1](https://link.zhihu.com/?target=https%3A//download.pytorch.org/whl/cu101/torch-1.3.1-cp37-cp37m-win_amd64.whl)
- [torchvison=0.4.2](https://link.zhihu.com/?target=https%3A//download.pytorch.org/whl/cu101/torchvision-0.4.2-cp37-cp37m-win_amd64.whl)
- python: 3.7
- cuda: 10.1
- pycocotools 

### 安装mmdetection
1. 创建虚拟环境：
```shell
conda create -n open-mmlab python=3.7
activate open-mmlab
```

2. 安装pytorch
```shell
pip install 下载目录/torch-1.3.1-cp37-cp37m-win_amd64.whl
pip install 下载目录/torchvision-0.4.2-cp37-cp37m-win_amd64.whl
```

3. torch源码修改
```shell
环境目录指open-mmlab虚拟环境位置
参考：https://zhuanlan.zhihu.com/p/159912450
a. 修改 环境目录下\Lib\site-packages\torch\include\c10\util\flat_hash_map.h
b. 修改 环境目录下\Lib\site-packages\torch\include\c10\util\order_preserving_flat_hash_map.h
c. 修改 环境目录下\Lib\site-packages\torch\utils\cpp_extension.py
d. 修改 环境目录下\Lib\site-packages\torch\include\torch\csrc\jit\argument_spec.h
e. 修改 环境目录下\Lib\site-packages\torch\include\pybind11\cast.h
```

4. pycocotools安装
```shell
git clone https://gitee.com/sqj7/cocoapi
cd cocoapi/PythonAPI
source open-mmlab
python setup.py build_ext --inplace
python setup.py build_ext install 
```

5. mmcv0.6.0安装
```shell
pip install mmcv==0.6.0
```
a. 修改环境目录下\Lib\site-packages\mmcv\utils找到config.py文件

如果pip安装不成功，那么使用源码安装
```shell
git clone https://gitee.com/sqj7/mmcv
#修改mmcv/utils找到config
cd mmcv
pip install -e .
```

6. mmdetection v2.1.0安装
```shell
git clone https://github.com/open-mmlab/mmdetection/tree/v2.1.0
pip install -r requirements.txt
python setup.py develop
```

### linux 版本安装
### 环境说明
- Linux
- Python 3.6+
- Pytorch 1.3+
- CUDA 9.2+
- GCC 5+
- [mmcv](https://github.com/open-mmlab/mmcv)

### 安装mmdetection
a. 创建虚拟环境：
```shell
conda create -n open-mmlab python=3.7
source activate open-mmlab
```

b. 安装pytorch
根据[pytorch website](https://pytorch.org/)安装pytorch和torchvision。
```shell
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

c. 克隆mmdetection
```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements.txt
python setup.py develop
```

###代码说明
```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   ├── cityscapes
│   │   ├── annotations
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012
├── code
│   ├── config 
│   ├── src 

```
├── code
│   ├── config  							#配置文件
│   │   ├── _base_  						#存放model,data,学习策略等配置信息
│   │   │   ├── cascade_rcnn_r50_fpn.py     #模型设置
│   │   │   ├── coco_detection.py			#训练数据路径设置，训练数据预处理设置
│   │   │   ├── default_runtime.py			#预训练模型路径设置
│   │   │   ├── schedule_1x.py				#学习率设置
│   │   ├── cascade_rcnn_r50_fpn_1x_coco.py	#设置_base_文件路径
│   │   ├── cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py #设置dcn fp16
```

train.py读取的config是cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py。

```
├── src                                    #源码
│   ├── creat_result.py 				   #创建模型预测结果
│   ├── Fabric2COCO.py					   #把数据格式转换为coco数据格式，方便网络读取
│   ├── model_transform.py				   #模型转换，把预训练模型转换为我们需要用的模型，主要该fc的classes数量
│   ├── test_for_mm.py					   #mmdetection的测试文件
```

##预训练模型准备

预训练模型可在README.md中进行查找下载
```
mmdetection
├── configs
│   ├── _base_
│   ├── xxx
│   ├── cascade_rcnn
│   ├── dcn 	
│   │   ├── README.md
```

Examples:
cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-2f1fca44
https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/dcn/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-2f1fca44.pth


##数据准备

训练数据需要转换为coco或者voc的标准格式
```
coco数据集合格式：
    def end_process(self):
        instance = {}
        instance['info'] = 'fabric_defect'
        instance['license'] = ['none']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        # self.save_coco_json(instance, os.path.join(os.getcwd(),  'data', self.new_dataset_dir,
        #                                            "annotations/" + 'instances_{}.json'.format(self.mode)))
        self.save_coco_json(instance, os.path.join(root,  'data', self.new_dataset_dir,
                                                   "annotations/" + 'instances_{}.json'.format(self.mode)))
```

详细代码见src/Fabric2COCO

##训练

训练代码在tool.train.py
```
mmdetection
├── tools
│   ├── train.py
```

ps: 训练是需要修改学习率(schedule_1x.py)  
学习率策略=0.00125*batchsize.    
batchsize = worders_per_gpu(多少个GPU) * samples_per_gpu(每个gpu多少张图片)

###单卡训练：
```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

EXAMPLE
```shell
python mmdetection/tools/train.py ../code/config/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py --gpus 1 --work-dir ../code/config/submit
```

###多卡训练：
```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

EXAMPLE
```shell
./tools/dist_train.sh ../code/config/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py 4 --work-dir ../code/config/submit
```



##测试
```shell
python mmdetection/tools/test.py ../code/config/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py ../code/config/submit/训练生成的权重 --out results.pkl --eval bbox
```

###Image demo
```shell
python demo/image_demo.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--score-thr ${SCORE_THR}]
```

EXAMPLE:
```shell
python demo/image_demo.py demo/demo.jpg configs/faster_rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth --device cpu
```


##可视化
###log分析

首先需要安装seaborn

pip install seaborn

```shell
python tools/analyze_logs.py plot_curve [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

EXAMPLE:

```shell
#分类loss
python tools/analyze_logs.py plot_curve log.json --keys loss_cls --legend loss_cls
#将loss保存到pdf
python tools/analyze_logs.py plot_curve log.json --keys loss_cls loss_bbox --out losses.pdf
#在同一张图中对比两次run的bbox map
python tools/analyze_logs.py plot_curve log1.json log2.json --keys bbox_mAP --legend run1 run2
#计算平均训练速度
python tools/analyze_logs.py cal_train_time log.json [--include-outliers]
#获取FLOPs和参数
python tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```