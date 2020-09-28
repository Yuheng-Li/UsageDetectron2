from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import random
import cv2


from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch



"""

This code can be used to train in default setting with coco format. 


Dataset: 

There are two commonly used classes MetadataCatalog and DatasetCatalog. Though I did not use them 
in the following code, I am still going to write down my understanding of them in case of future need.

An instance of MetadataCatalog contains some high-level idea of this dataset. Such as path to images 
number of classes, their name, etc. It does not have information of each image and instances. 
DatasetCatalog, on the other hand, is a list with the lenght of total training/test data. Each item inside
is a dict, and this dict contains all information of this image. In my instance segmentation example, keys 
of each dict are: ['file_name', 'height', 'width', 'image_id', 'annotations']

If you use coco format data, you can first regist them through code: (for example, we use ade20k)
register_coco_instances("ade20k(This is just a name)", {}, "path_to_ade20k_json", "path_to_images") 
and later you can get metadata or dataset by 
ade20k_metadata = MetadataCatalog.get( "ade20k(the name you used in register)" )
ade20k_dict = DatasetCatalog.get( "ade20k(the name you used in register)" )

"""



def main(args):

    # first regiest dataset I will use 
    register_coco_instances("my_dataset_train", {}, "training.json", "../data/ade20k/full_data/images/training/")
    register_coco_instances("my_dataset_val", {}, "validation.json", "../data/ade20k/full_data/images/validation/")

    # this is just a default cfg files 
    cfg = get_cfg()
    # accordinig to different yaml file, it will change cfg files accordiningly 
    cfg.merge_from_file("../detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")

    # This is some task specific changes I made for training ade20k dataset 
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml") 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 150  # 150 classes 
    cfg.SOLVER.IMS_PER_BATCH = 16 #  this is default one


    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # I highly suggest read source code of DefaultTrainer again, if you forget why you did this. 
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()



if __name__ == "__main__":

    # You should chech source code of default_argument_parser() to see what args are there (e.g., num-gpus etc)
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    
    # This one is used to launch main function, distributed training is called inside launch 
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
