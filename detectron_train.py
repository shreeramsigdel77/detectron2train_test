import os
import numpy as np
import json
import itertools
import random

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.data.datasets import register_coco_instances
from logger import logger



#coco-data path
#coco_data_path = "/home/pasonatech/Downloads/crescent_dataset/"

# coco_data_path = "/home/pasonatech/UEoutputs/8/type1_v2_train/train_bolt_roi"

# coco_data_path = "/home/pasonatech/detectron/detectron2/gbox/crescent_dataset_randomcolor"

# coco_data_path = "/home/pasonatech/Desktop/10/10_9/crescent_blender_ram"

#blend proc dataset
coco_data_path = "/home/pasonatech/blender_proc/BlenderProc-master/examples/crescent_test/output_randCrescent/coco_data"

register_coco_instances(
    name = "marker",
    metadata = {},
    json_file = f'{coco_data_path}/coco_annotations.json',
    image_root = f'{coco_data_path}',
)

MetadataCatalog.get("marker").thing_classes = ['1']
abc_metadata_train = MetadataCatalog.get("marker")
logger.purple(abc_metadata_train)
dataset_dicts = DatasetCatalog.get("marker")
logger.blue(dataset_dicts)

#fine tuning
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.model_zoo import get_config_file, get_checkpoint_url

#model_config_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
model_config_path = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"


cfg = get_cfg()
cfg.merge_from_file(get_config_file(model_config_path))
cfg.DATASETS.TRAIN = ("marker",)
cfg.DATASETS.TEST = ()   
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = get_checkpoint_url(model_config_path)  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.0025
cfg.SOLVER.MAX_ITER = 35000  # iterations train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Number of class 
cfg.INPUT.MIN_SIZE_TRAIN = (1024,) #bydeafult 800
cfg.INPUT.MAX_SIZE_TRAIN = 1024


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=True)
trainer.train()




