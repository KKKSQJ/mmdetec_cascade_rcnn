import torch


def model_transform_for_v1(model_path, num_classes):
    # gen coco pretrained weight
    #num_classes = 21
    # model_coco = torch.load("../../checkpoints/cascade_rcnn_r50_fpn_1x_20190501-3b6211ab.pth")
    num_classes += 1  #v1需要再加一类（背景类）
    model_coco = torch.load(model_path)
    # weight
    model_coco["state_dict"]["bbox_head.0.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.0.fc_cls.weight"][
                                                            :num_classes, :]
    model_coco["state_dict"]["bbox_head.1.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.1.fc_cls.weight"][
                                                            :num_classes, :]
    model_coco["state_dict"]["bbox_head.2.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.2.fc_cls.weight"][
                                                            :num_classes, :]
    # bias
    model_coco["state_dict"]["bbox_head.0.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.0.fc_cls.bias"][
                                                          :num_classes]
    model_coco["state_dict"]["bbox_head.1.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.1.fc_cls.bias"][
                                                          :num_classes]
    model_coco["state_dict"]["bbox_head.2.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.2.fc_cls.bias"][
                                                          :num_classes]
    # save new model
    torch.save(model_coco, "../../checkpoints/cascade_rcnn_r50_coco_pretrained_weights_classes_%d.pth" % num_classes)

def model_transform_for_v2(model_path ,num_classes):
    # gen coco pretrained weight
    #num_classes = 21
    # model_coco = torch.load("../../checkpoints/cascade_rcnn_r50_fpn_1x_20190501-3b6211ab.pth")
    num_classes += 1
    model_coco = torch.load(model_path)
    # weight
    model_coco["state_dict"]["roi_head.bbox_head.0.fc_cls.weight"] = model_coco["state_dict"]["roi_head.bbox_head.0.fc_cls.weight"][
                                                            :num_classes, :]
    model_coco["state_dict"]["roi_head.bbox_head.1.fc_cls.weight"] = model_coco["state_dict"]["roi_head.bbox_head.1.fc_cls.weight"][
                                                            :num_classes, :]
    model_coco["state_dict"]["roi_head.bbox_head.2.fc_cls.weight"] = model_coco["state_dict"]["roi_head.bbox_head.2.fc_cls.weight"][
                                                            :num_classes, :]
    # bias
    model_coco["state_dict"]["roi_head.bbox_head.0.fc_cls.bias"] = model_coco["state_dict"]["roi_head.bbox_head.0.fc_cls.bias"][
                                                          :num_classes]
    model_coco["state_dict"]["roi_head.bbox_head.1.fc_cls.bias"] = model_coco["state_dict"]["roi_head.bbox_head.1.fc_cls.bias"][
                                                          :num_classes]
    model_coco["state_dict"]["roi_head.bbox_head.2.fc_cls.bias"] = model_coco["state_dict"]["roi_head.bbox_head.2.fc_cls.bias"][
                                                          :num_classes]
    # save new model
    torch.save(model_coco, "../../checkpoints/cascade_rcnn_r50_dcn_coco_pretrained_weights_classes_%d.pth" % num_classes)
if __name__ == "__main__":
    #预训练模型转换，需要将COCO数据集转换为当前数据集可用类型
    model_path = '../../checkpoints/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-2f1fca44.pth'
    num_classes = 20
    #model_transform_for_v1(model_path, num_classes)
    model_transform_for_v2(model_path, num_classes)
    print("----->ok<-------")

