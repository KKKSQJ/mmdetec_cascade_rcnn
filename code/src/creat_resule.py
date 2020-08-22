import time, os
import json
# import mmcv
from mmdet.apis import init_detector, inference_detector
defect_name2label = {
    '破洞': 1, '水渍': 2, '油渍': 2, '污渍': 2, '三丝': 3, '结头': 4, '花板跳': 5, '百脚': 6, '毛粒': 7, '粗经': 8,
    '松经': 9, '断经': 10, '吊经': 11, '粗维': 12, '纬缩': 13, '浆斑': 14, '整经结': 15, '星跳': 16, '跳花': 16,
    '断氨纶': 17, '稀密档': 18, '浪纹档': 18, '色差档': 18, '磨痕': 19, '轧痕': 19, '修痕': 19, '烧毛痕': 19, '死皱': 20,
    '云织': 20, '双纬': 20, '双经': 20, '跳纱': 20, '筘路': 20, '纬纱不良': 20,
}

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--wight-path', default='', type=str, help='path to the dataset')
args = parser.parse_args()

def result2json(checkpoint_file):
    config_file = 'code/cascade_skypool_dataenhancement.py'  # 修改成自己的配置文件
    # checkpoint_file = 'work_dirs/cascade_custom_data_enhancement/epoch_56.pth'  # 修改成自己的训练权重

    test_path = 'data/guangdong1_round1_testA_20190818/testA'  # 官方测试集图片路径

    json_name = "result_" + "" + time.strftime("%Y%m%d%H%M%S", time.localtime()) + ".json"

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    img_list = []
    for img_name in os.listdir(test_path):
        if img_name.endswith('.jpg'):
            img_list.append(img_name)

    result = []
    for i, img_name in enumerate(img_list, 1):
        full_img = os.path.join(test_path, img_name)
        predict = inference_detector(model, full_img)
        for i, bboxes in enumerate(predict, 1):
            if len(bboxes) > 0:
                # defect_label = lable[i]
                # print(defect_label)
                defect_label = i
                print(i)
                image_name = img_name
                for bbox in bboxes:
                    x1, y1, x2, y2, score = bbox.tolist()
                    x1, y1, x2, y2 = round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)  # save 0.00
                    result.append(
                        {'name': image_name, 'category': defect_label, 'bbox': [x1, y1, x2, y2], 'score': score})

    with open(json_name, 'w') as fp:
        json.dump(result, fp, indent=4, separators=(',', ': '))


if __name__ == "__main__":
    new_lable2name = {v: k for k, v in defect_name2label.items()}
    result2json(args.wight_path)

