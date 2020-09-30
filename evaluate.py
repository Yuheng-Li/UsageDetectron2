from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import os
import pickle


def get_model(cfg):
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    return model 


def get_config():
    cfg = get_cfg()
    cfg.merge_from_file("/home/yuheng/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES= 150
    return cfg 



def fire(path):

    # find all models inside of this path (end with .pth)
    models_path = []
    for file in os.listdir(path):
        if file.endswith(".pth"):
            models_path.append( os.path.join(path,file) )


    # prepare evaluator and loader  
    cfg = get_config()
    evaluator = COCOEvaluator("ade_val", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "ade_val")

    # eval all models 
    compare_dict = {}
    for model_path in models_path:    
        cfg.MODEL.WEIGHTS = model_path
        model = get_model(cfg)
        results = inference_on_dataset(  model, val_loader, evaluator)

        AP = results['segm']['AP']
        compare_dict[model_path] = AP

    # sort the dict accordiing to value 
    compare_result = sorted(compare_dict.items(), key=lambda x: x[1], reverse=True)

    # save and print results
    with open("compare_result.txt", "wb") as fp:   
        pickle.dump(compare_result, fp)

    for item in compare_result:
        print( item )




if __name__ == '__main__':

    register_coco_instances("ade_val", {}, "validation.json", "../ADE20K_2016_07_26/full_data/images/validation/")
 
    path_to_models_folder = 'temp'
    
    fire(path_to_models_folder)