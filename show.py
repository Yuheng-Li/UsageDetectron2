from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import random
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import os 
from PIL import Image
from detectron2.modeling import build_model


# get config file 
cfg = get_cfg()
cfg.merge_from_file("/home/yuheng/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
cfg.MODEL.WEIGHTS = 'model_final.pth'
cfg.MODEL.WEIGHTS
cfg.MODEL.ROI_HEADS.NUM_CLASSES= 150


# get metadata used by visualizer
register_coco_instances("ade", {}, "validation.json", "../ADE20K_2016_07_26/full_data/images/validation/")
ade_metadata = MetadataCatalog.get('ade')
no_use = DatasetCatalog.get("ade")



predictor = DefaultPredictor(cfg)


path = '../ADE20K_2016_07_26/full_data_bedroom/images/validation/'
files = os.listdir(path)
files.sort()

for i, file in enumerate(files):
    im = cv2.imread(   os.path.join(  path, file )  )
    outputs = predictor(im)  

    v = Visualizer(im[:, :, ::-1], metadata=ade_metadata, scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    out_img = out.get_image()[:, :, ::-1]
    Image.fromarray(out_img).save(str(i).zfill(4)+'.png')


