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



vid_out_name = "/home/pasonatech/Desktop/10/10_10/inference_video/from_3dflow/mov1.mp4"
#path_vid = "/home/pasonatech/Desktop/trash_video/3.mov"
path_vid = "/home/pasonatech/Desktop/10/10_10/testvideo/IMG_5567.MOV"

#coco-data path


#300 3dflowimages
coco_data_path = "/home/pasonatech/UEoutputs/9/cresecent_3dflow/coco-files/crescent_dataset"
json_path = f'{coco_data_path}/output.json'

img_path = f'{coco_data_path}'



###############################################

register_coco_instances(

    name="marker",

    metadata={},

    json_file=json_path,

    image_root=img_path,

)

MetadataCatalog.get("marker").thing_classes = ['cresecent']

box_metadata = MetadataCatalog.get("marker")
model_config_path = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"

###############################################
cfg = get_cfg()

cfg.merge_from_file(get_config_file(model_config_path))

cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = get_checkpoint_url(model_config_path)  # Let training initialize from model zoo

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # one class (garbage = 3)

cfg.INPUT.MIN_SIZE_TEST = 1024
cfg.INPUT.MAX_SIZE_TEST = 1024

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

##########################################################
#texture cresecent weight file
model_path ="/home/pasonatech/detectron/detectron2/gbox/output_cresecent300img/"

cfg.MODEL.WEIGHTS = os.path.join(model_path, "model_final.pth")


#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0029999.pth")

# set the testing threshold for this model

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
cfg.DATASETS.TEST = ("marker", )
predictor = DefaultPredictor(cfg)
dataset_dicts = DatasetCatalog.get("marker")

cap = cv2.VideoCapture(path_vid)

fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))



out = cv2.VideoWriter(vid_out_name, fourcc, 20.0,(frame_width,frame_height),True )

# print(int(cap.get(3)))
# print(int(cap.get(4)))


while(cap.isOpened()):
    ret,frame = cap.read()
    if ret == True:
        outputs = predictor(frame)
        #v = Visualizer(ima[:, :, ::-1],metadata=box_metadata,scale=0.8,instance_mode=ColorMode.IMAGE_BW)
        v = Visualizer(frame[:, :, ::-1],metadata=box_metadata,scale=1)   # remove the colors of unsegmented pixels
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        imga = v.get_image()[:, :, ::-1]
        
        #print(imga.shape)
        
        cv2.imshow('Frame', imga)
        out.write(imga)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()

cv2.destroyAllWindows()




