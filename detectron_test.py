import json
import os
import random
import cv2
import numpy as np
import glob

from PIL import Image
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.utils.visualizer import Visualizer,ColorMode
from detectron2.config import get_cfg
from detectron2.model_zoo import get_config_file, get_checkpoint_url

test_image_list = []

test_data_path = "/home/pasonatech/Desktop/10/10_7/test_data/net_data_model/*"

coco_data_path = "/home/pasonatech/blender_proc/BlenderProc-master/examples/crescent_test/collage_merged_img"




json_path = f'{coco_data_path}/coco_annotations.json'

img_path = f'{coco_data_path}'



###############################################

register_coco_instances(

    name="marker",

    metadata={},

    json_file=json_path,

    image_root=img_path,

)

MetadataCatalog.get("marker").thing_classes = ['1']

box_metadata = MetadataCatalog.get("marker")
model_config_path = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
#model_config_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
###############################################
cfg = get_cfg()

cfg.merge_from_file(get_config_file(model_config_path))

cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = get_checkpoint_url(model_config_path)  # Let training initialize from model zoo

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # one class (markedbolt=1)

cfg.INPUT.MIN_SIZE_TEST = 512
cfg.INPUT.MAX_SIZE_TEST = 512
#cfg.INPUT.MIN_SIZE_TEST = 0 #Size of the smallest side of the image during testing. Set to zero to disable resize in testing

#cfg.INPUT.MAX_SIZE_TEST = 1333  # Maximum size of the side of the image during testing by deafult 1333
# Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
#cfg.INPUT.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
#cfg.INPUT.MAX_SIZE_TEST = 1333


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

##########################################################
#texture cresecent weight file


model_path = "output/"
cfg.MODEL.WEIGHTS = os.path.join(model_path, "model_final.pth")



cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
cfg.DATASETS.TEST = ("marker", )
predictor = DefaultPredictor(cfg)
dataset_dicts = DatasetCatalog.get("marker")


save_img_path = "/home/pasonatech/Desktop/10/10_21/blendProc/inference_results/real_img/"

h.join(save_img_path,'predicted.jpg'),imga)
# cv2.waitKey(1000)

i = 0
for filename in glob.glob(test_data_path):
    ima = cv2.imread(filename)
    #ima = cv2.resize(ima,(1024,1024))
    outputs = predictor(ima)
    #print(f"Outputs {outputs}")
    #v = Visualizer(ima[:, :, ::-1],metadata=box_metadata,scale=0.8,instance_mode=ColorMode.IMAGE_BW)
    v = Visualizer(ima[:, :, ::-1],metadata=box_metadata,scale=1,instance_mode=ColorMode.SEGMENTATION)   # remove the colors of unsegmented pixels
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    imga = v.get_image()[:, :, ::-1]
    cv2.imshow("predictions1", imga)
    cv2.imwrite(os.path.join(save_img_path,str(i)+'.jpg'),imga)
    k = cv2.waitKey(1000) & 0xff
    if k == 27:
        break
    i+=1

cv2.destroyAllWindows()



