"""@javis"""
import os
import json
import numpy as np
import shutil
import pandas as pd
from tqdm import tqdm


defect_name2label = {
    '破洞': 1, '水渍': 2, '油渍': 2, '污渍': 2, '三丝': 3, '结头': 4, '花板跳': 5, '百脚': 6, '毛粒': 7, '粗经': 8,
    '松经': 9, '断经': 10, '吊经': 11, '粗维': 12, '纬缩': 13, '浆斑': 14, '整经结': 15, '星跳': 16, '跳花': 16,
    '断氨纶': 17, '稀密档': 18, '浪纹档': 18, '色差档': 18, '磨痕': 19, '轧痕': 19, '修痕': 19, '烧毛痕': 19, '死皱': 20,
    '云织': 20, '双纬': 20, '双经': 20, '跳纱': 20, '筘路': 20, '纬纱不良': 20,
}
# defect_name2label = {
#     '破洞': 1, '水渍': 2, '油渍': 3, '污渍': 4, '三丝': 5, '结头': 6, '花板跳': 7, '百脚': 8, '毛粒': 9,
#     '粗经': 10, '松经': 11, '断经': 12, '吊经': 13, '粗维': 14, '纬缩': 15, '浆斑': 16, '整经结': 17, '星跳': 18, '跳花': 19,
#     '断氨纶': 20, '稀密档': 21, '浪纹档': 22, '色差档': 23, '磨痕': 24, '轧痕': 25, '修痕': 26, '烧毛痕': 27, '死皱': 28, '云织': 29,
#     '双纬': 30, '双经': 31, '跳纱': 32, '筘路': 33, '纬纱不良': 34,
# }


class Fabric2COCO:

    def __init__(self, _new_dataset_dir,root, mode):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.mode = mode
        self.root = root
        self.new_dataset_dir = _new_dataset_dir
        self._init_categories()
        if not os.path.exists(os.path.join(root, 'data', self.new_dataset_dir, "{}".format(self.mode))):
            os.makedirs(os.path.join(root, 'data', self.new_dataset_dir, "{}".format(self.mode)))
        if not os.path.exists(os.path.join(root, 'data', self.new_dataset_dir, "{}".format('annotations'))):
            os.makedirs(os.path.join(root, 'data', self.new_dataset_dir, "{}".format('annotations')))

    def to_coco(self, anno_file, img_dir):
        anno_result = pd.read_json(open(anno_file, "r"))
        name_list = anno_result["name"].unique()
        for i, img_name in enumerate(tqdm(name_list)):
            # creat val dataset
            if i % 10:
                if self.mode == 'val':
                    continue
            else:
                if self.mode == 'train':
                    continue
            img_anno = anno_result[anno_result["name"] == img_name]
            bboxs = img_anno["bbox"].tolist()
            defect_names = img_anno["defect_name"].tolist()
            assert img_anno["name"].unique()[0] == img_name

            img_path = os.path.join(img_dir, img_name)
            # img = cv2.imread(img_path)
            # h,w,c=img.shape
            h, w = 1000, 2446
            self.images.append(self._image(img_path, h, w))

            self._cp_img(img_path)

            for bbox, defect_name in zip(bboxs, defect_names):
                label = defect_name2label[defect_name]
                annotation = self._annotation(label, bbox)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1

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

    def _init_categories(self):
        for v in range(1, 21):
            print(v, end=' ')
            category = {}
            category['id'] = v
            category['name'] = str(v)
            category['supercategory'] = 'defect_name'
            self.categories.append(category)
        # for k, v in defect_name2label.items():
        #     category = {}
        #     category['id'] = v
        #     category['name'] = k
        #     category['supercategory'] = 'defect_name'
        #     self.categories.append(category)

    def _image(self, path, h, w):
        image = {}
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path)
        return image

    def _annotation(self, label, bbox):
        area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        points = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = label
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = area
        return annotation

    def _cp_img(self, img_path):
        # shutil.copy(img_path, os.path.join(os.getcwd(),  'data', self.new_dataset_dir,
        #                                    self.mode + "/{}".format(os.path.basename(img_path))))
        shutil.copy(img_path, os.path.join(self.root,  'data', self.new_dataset_dir,
                                           self.mode + "/{}".format(os.path.basename(img_path))))

    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        '''coco,[x,y,w,h]'''
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    def save_coco_json(self, instance, save_path):
        with open(save_path, 'w') as fp:
            json.dump(instance, fp, indent=1, separators=(',', ': '))

if __name__ == '__main__':
    new_dataset_dir = 'skypool_first_trainval'
    cwd = os.getcwd().split(os.sep)
    root = r'D:\dataset'  #训练数据输出路径
    dataroot = r'F:\DataSet\guangdong' #原始数据路径
    # if not os.path.exists(os.path.join(os.getcwd(), 'data', new_dataset_dir)):
    #     os.makedirs(os.path.join(os.getcwd(), 'data', new_dataset_dir))
    if not os.path.exists(os.path.join(root, 'data', new_dataset_dir)):
        os.makedirs(os.path.join(root, 'data', new_dataset_dir))
    '''转换有瑕疵的样本为coco格式'''

    train1_images = os.path.join(dataroot,"guangdong1_round1_train1_20190818","defect_Images")#"./data/guangdong1_round1_train1_20190818/defect_Images"
    train1_anno = os.path.join(dataroot,"guangdong1_round1_train1_20190818","Annotations","anno_train.json")#"./data/guangdong1_round1_train1_20190818/Annotations/anno_train.json"
    train2_images = os.path.join(dataroot,"guangdong1_round1_train2_20190828","defect_Images")#"./data/guangdong1_round1_train2_20190828/defect_Images"
    train2_anno = os.path.join(dataroot,"guangdong1_round1_train2_20190828","Annotations","anno_train.json")#"./data/guangdong1_round1_train2_20190828/Annotations/anno_train.json"
    '''生成训练集'''
    fabric2coco_train = Fabric2COCO(new_dataset_dir, root, mode='train')
    fabric2coco_train.to_coco(train1_anno, train1_images)
    fabric2coco_train.to_coco(train2_anno, train2_images)
    fabric2coco_train.end_process()
    '''生成验证集'''
    fabric2coco_val = Fabric2COCO(new_dataset_dir,root, mode='val')
    fabric2coco_val.to_coco(train1_anno, train1_images)
    fabric2coco_val.to_coco(train2_anno, train2_images)
    fabric2coco_val.end_process()







