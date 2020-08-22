# mmdetec_cascade_rcnn
该代码算法：cascade_rcnn_r50fpn_dconv_c3-c5_1x_coco。baseline:mmdetectionV2.0+

## 安装
### Windows版本安装
参考网址：https://zhuanlan.zhihu.com/p/159912450

### 环境说明
- windwos
- vs:2015 or 2017
- mmdetection:2.1.0
- mmcv:0.6.0
- torch=1.3.1  https://link.zhihu.com/?target=https%3A//download.pytorch.org/whl/cu101/torch-1.3.1-cp37-cp37m-win_amd64.whl
- torchvison=0.4.2 https://link.zhihu.com/?target=https%3A//download.pytorch.org/whl/cu101/torchvision-0.4.2-cp37-cp37m-win_amd64.whl
- python:3.7
- cuda:10.1
- pycocotools 

### 安装mmdetection
1. 创建虚拟环境：
'''shell
conda create -n open-mmlab python=3.7
activate open-mmlab
'''

2. 安装pytorch
'''shell
pip install 下载目录/torch-1.3.1-cp37-cp37m-win_amd64.whl
pip install 下载目录/torchvision-0.4.2-cp37-cp37m-win_amd64.whl
'''

3. torch源码修改
环境目录指open-mmlab虚拟环境位置
参考：https://zhuanlan.zhihu.com/p/159912450
a. 修改 环境目录下\Lib\site-packages\torch\include\c10\util\flat_hash_map.h
b. 修改 环境目录下\Lib\site-packages\torch\include\c10\util\order_preserving_flat_hash_map.h
c. 修改 环境目录下\Lib\site-packages\torch\utils\cpp_extension.py
d. 修改 环境目录下\Lib\site-packages\torch\include\torch\csrc\jit\argument_spec.h
e. 修改 环境目录下\Lib\site-packages\torch\include\pybind11\cast.h

4. pycocotools安装
'''shell
git clone https://gitee.com/sqj7/cocoapi
cd cocoapi/PythonAPI
source open-mmlab
python setup.py build_ext --inplace
python setup.py build_ext install 
'''

5. mmcv0.6.0安装
'''shell
pip install mmcv==0.6.0
'''
a. 修改环境目录下\Lib\site-packages\mmcv\utils找到config.py文件

如果pip安装不成功，那么使用源码安装
'''shell
git clone https://gitee.com/sqj7/mmcv
#修改mmcv/utils找到config
cd mmcv
pip install -e .
'''

6. mmdetection v2.1.0安装
'''shell
git clone https://github.com/open-mmlab/mmdetection/tree/v2.1.0
pip install -r requirements.txt
python setup.py develop
'''
