from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import random
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import os 
from PIL import Image
import argparse
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import detectron2.data.transforms as T
import torch
import numpy as np


def get_my_config():
    """
    You should modify this function according to what config you used for training 

    The code here is for when I use X101-FPN for ade20k dataset 

    """
    cfg = get_cfg()
    cfg.merge_from_file("/home/yuheng/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
    cfg.MODEL.WEIGHTS = 'temp/model_0079999.pth'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES= 150

    return cfg 


def prepare_input_images():
    """
    You should modify this function according to what data you want to input 

    This code should return a list which includes all inference images path 
    """

    path = '../ADE20K_2016_07_26/full_data_bedroom/images/validation/'
    files = os.listdir(path)
    files = [  os.path.join(path, file) for file in files  ]
    files.sort()

    return files




def get_metadataset():
    """
    You should modify this function according to own needs

    Ideadly this function is not necessary for inference, but it might be useful when you have 
    metadata when you do process outputs (As you probably want to know the name of predicted class etc)
    If you do not need this function then just return None 
    """
    from detectron2.data.datasets import register_coco_instances
    register_coco_instances("ade_metadata", {}, "validation.json", "../ADE20K_2016_07_26/full_data/images/validation/")
    ade_metadata = MetadataCatalog.get("ade_metadata")
    no_use = DatasetCatalog.get("ade_metadata") # I do not why, you have to call this, otherwise ade_metadata is not complete 

    return ade_metadata






def process_outputs(  image_files, outputs, metadata ):
    """
    You should modify this function according to own needs
    outputs is output from detectron2 model and batch_data is coresponding input images path (in a list)
    metadata is output from get_metadataset() 

    For example, if you need to save outputs into an image then you should write code for saving 
    """

    for image_file, output in zip(image_files, outputs):

        basename = os.path.basename(image_file)[:-4] 
        output = output['instances'].get_fields() # it has 4 keys here: 'pred_boxes', 'scores', 'pred_classes', 'pred_masks'

        total_detected_instances = len(output['scores'])


        if total_detected_instances > 0: # otherwise no detected instance

            sem = torch.zeros_like(  output['pred_masks'][0]  ).float()
            ins = torch.zeros_like(  output['pred_masks'][0]  ).float()

            current_instance = 0 
            for i in range( total_detected_instances-1, -1, -1 ): # backwards indexing (lowest score first). 'scores' are sorted from detectron2
                
                # these two are used to create sem and ins map
                current_class = output['pred_classes'][i] + 1 # I set the first class is 1 for ade, whereas in detectron2 they shift classes by 1 
                current_instance += 1 

                sem.masked_fill_( output['pred_masks'][i], current_class )
                ins.masked_fill_( output['pred_masks'][i], current_instance )

            # we assume there are only 255 instances at most in one iamge 
            ins[ins>255] = 255 

            # then save sem and ins using basename 
            ins = Image.fromarray( np.array(ins.cpu()).astype('uint8') )
            sem = Image.fromarray( np.array(sem.cpu()).astype('uint8') )

            sem.save(  'annotation/' + basename+'.png' )
            ins.save(  'annotation_instance/' + basename+'.png' )
        













#######################################################################################################


class Predictor:

    " This is similar to detectron2.engine.DefaultPredictor. I made it support batch inference "

    def __init__(self, cfg):

        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()

        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge( [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format


    def __call__(self, original_images):
        """
        Args:
            original_images: path list.

        Returns:
            predictions: list of dict:

        """
        original_images = [ cv2.imread(path)  for path in original_images ]
  
        if self.input_format == "RGB":
            original_images = [ img[:, :, ::-1]  for img in original_images ]


        batch_inputs = [] 
        for  original_image in original_images:
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            batch_inputs.append(inputs)

        with torch.no_grad():  
            predictions = self.model( batch_inputs ) 
            return predictions



def create_dataloader(image_files, batch_size):
    "Just group list into chuncks to mimic dataloader"
    total_len = len(image_files)
    image_files = [ image_files[i:i+batch_size] for i in range(0, total_len, batch_size) ] 
    return image_files




def fire(batch_size):

 
    cfg = get_my_config()
    metadata = get_metadataset()



    predictor = Predictor(cfg)

    image_files = prepare_input_images()
    dataloader = create_dataloader(image_files, batch_size)

    for batch_data in dataloader:
        outputs = predictor(batch_data)
        process_outputs( batch_data, outputs, metadata )



if __name__ == "__main__":

    fire(batch_size=2)

