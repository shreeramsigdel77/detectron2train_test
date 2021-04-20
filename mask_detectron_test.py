import json
import os
import random
import cv2
import numpy as np
import glob
import cloudpickle as cPickle

from PIL import Image
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.utils.visualizer import Visualizer,ColorMode
from detectron2.config import get_cfg
from detectron2.model_zoo import get_config_file, get_checkpoint_url

test_image_list = []
#test_data_path = "/home/pasonatech/Desktop/trash_video/images/*.jpg"
#test_data_path = "/home/pasonatech/aug_real_combine/aug_sim_com_garbage/*.png"
#test_data_path = "/home/pasonatech/Desktop/exit_test_img/*.jpg"
#test_data_path = "/home/pasonatech/Desktop/trash_real_img/*.jpg"
#test_data_path = "/home/pasonatech/Desktop/infer_results/1.jpg"
#test_data_path = "/home/pasonatech/Desktop/ram/sekisui/solarpanel/3-2.太陽光電池モジュールの固定（3720）-20200605T082019Z-001/3-2.太陽光電池モジュールの固定（3720）/合格/*.jpg"
#test_data_path = "/home/pasonatech/Desktop/ram/sekisui/solarpanel/3-2.太陽光電池モジュールの固定（3720）-20200605T082019Z-001/3-2.太陽光電池モジュールの固定（3720）/不合格/シャープ以外は撮影不要/*.jpg"


#one test pic
test_data_path = "/home/pasonatech/Desktop/7/7_27/test_pic/*.jpg"
#good real image 2
#test_data_path = "/home/pasonatech/Desktop/ram/sekisui/solarpanel/3-1.太陽光電池モジュールの固定（3719）-20200605T003621Z-001/3-1.太陽光電池モジュールの固定（3719）/合格/*.jpg"


#test_data_path = "/home/pasonatech/Desktop/7/7_27/sim_testdata/*.png" 



#test_data_path = "/home/pasonatech/Desktop/7/7_9/2bolt_with_rand_color/sim_testdata/*.png"


#coco-data path

coco_data_path = "/home/pasonatech/Desktop/7/7_27/combine-coco"    #511image
#coco_data_path = "/home/pasonatech/Desktop/7/7_15/coco-data/combine1"  #561 image (50 v shap image combined)

#coco_data_path = "/home/pasonatech/Desktop/7/7_15/coco-data/combine2" # 750 image (250 v mark image combined)


#coco_data_path ="/home/pasonatech/Desktop/7/7_6/2bolt_in_1frame_adj/coco-data/distractor/type1"
json_path = f'{coco_data_path}/HSR-coco.json'

img_path = f'{coco_data_path}'



###############################################

register_coco_instances(

    name="marker",

    metadata={},

    json_file=json_path,

    image_root=img_path,

)

MetadataCatalog.get("marker").thing_classes = ['bolt-roi']

box_metadata = MetadataCatalog.get("marker")
model_config_path = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
#model_config_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
###############################################
cfg = get_cfg()

cfg.merge_from_file(get_config_file(model_config_path))

cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = get_checkpoint_url(model_config_path)  # Let training initialize from model zoo

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # one class (markedbolt=1)

cfg.INPUT.MIN_SIZE_TEST = 1024
cfg.INPUT.MAX_SIZE_TEST = 1024
#cfg.INPUT.MIN_SIZE_TEST = 0 #Size of the smallest side of the image during testing. Set to zero to disable resize in testing

#cfg.INPUT.MAX_SIZE_TEST = 1333  # Maximum size of the side of the image during testing by deafult 1333
# Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
#cfg.INPUT.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
#cfg.INPUT.MAX_SIZE_TEST = 1333


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

##########################################################

model_path ="/home/pasonatech/detectron/detectron2/gbox/output7_27_distractorobject/"
cfg.MODEL.WEIGHTS = os.path.join(model_path, "model_final.pth")

#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0009999.pth")

# set the testing threshold for this model

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
cfg.DATASETS.TEST = ("marker", )
predictor = DefaultPredictor(cfg)
dataset_dicts = DatasetCatalog.get("marker")


#save_img_path = "/home/pasonatech/Desktop/7/7_17/infer_result/real_image30k/"

save_img_path = "/home/pasonatech/Desktop/7/7_27/infer_result/test_image/"
# ima = cv2.imread(test_data_path)
# outputs = predictor(ima)
# v = Visualizer(ima[:, :, ::-1],metadata=box_metadata,scale=0.8,instance_mode=ColorMode.IMAGE_BW)   # remove the colors of unsegmented pixels
# v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# imga = v.get_image()[:, :, ::-1]
# cv2.imshow("predictions1", imga)
# cv2.imwrite(os.path.join(save_img_path,'predicted.jpg'),imga)
# cv2.waitKey(1000)

i = 0
for filename in glob.glob(test_data_path):
    ima = cv2.imread(filename)
    #ima = cv2.resize(ima,(1024,1024))
    outputs = predictor(ima)
    #v = Visualizer(ima[:, :, ::-1],metadata=box_metadata,scale=0.8,instance_mode=ColorMode.IMAGE_BW)
    v = Visualizer(ima[:, :, ::-1],metadata=box_metadata,scale=1,instance_mode=ColorMode.SEGMENTATION)   # remove the colors of unsegmented pixels
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #coordinates
    bounding_box = outputs["instances"].pred_boxes
    bounding_box_class = outputs["instances"].pred_classes
    print(bounding_box) 
    print(bounding_box_class)
   #a = *bounding_box
    #print(*bounding_box)
    bounding_box = str(bounding_box)
    a = bounding_box[15:-21]
    a = a.replace(",",'')
    b = a.split()
    print(a)
    
    x1=round(float(b[0]))
    y1=round(float(b[1]))
    x2=round(float(b[2]))
    y2=round(float(b[3]))
    
    imga = v.get_image()[:, :, ::-1]
    
    im_crop =imga[y1:y2,x1:x2]
    
    cv2.imshow("Crop", im_crop)
    cv2.imwrite(os.path.join(save_img_path,str(i)+'crop.jpg'),im_crop)

    #finding contours(a curve joining all the continuous points (along the boundary), having same color or intensity)
    #tARGET to have each pixel either black or white
    #convert bgr to RGB
    img_rgb = cv2.cvtColor(im_crop, cv2.COLOR_BGR2RGB)
    
    #test
    cv2. imshow("Test1",img_rgb)
    cv2.imwrite(os.path.join(save_img_path,str(i)+'rgb1.jpg'),img_rgb)
    k = cv2.waitKey(1000) & 0xff
    if k == 27:
        break

    #convert to grayscale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    #convert to grayscale
    #img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV_FULL)

    #test2
    cv2. imshow("Test2",img_gray)
    cv2.imwrite(os.path.join(save_img_path,str(i)+'gray2.jpg'),img_gray)
    k = cv2.waitKey(1000) & 0xff
    if k == 27:
        break

    #creating binary threshhold image
    _, binary = cv2.threshold(img_gray, 127,255, cv2.THRESH_BINARY_INV)

    binary  = cv2.bitwise_not(binary)
    #Preview
    #test3
    cv2. imshow("Test3",binary)
    cv2.imwrite(os.path.join(save_img_path,str(i)+'binary3.jpg'),binary)
    k = cv2.waitKey(1000) & 0xff
    if k == 27:
        break
    
    #finding the contours from the thresholded image
    contours, hierarchy = cv2.findContours(
        image = binary,
        mode = cv2.RETR_TREE,
        method = cv2.CHAIN_APPROX_SIMPLE
    )

    

    # cont_image = cv2.drawContours(
    #     image = binary,
    #     contours = contours,
    #     contourIdx = -1,
    #     color = (0,255,0),
    #     thickness=2,
    #     lineType= None,
    #     hierarchy=None,
    #     maxLevel=None,
    #     offset=None
    # )
    im_crop = im_crop.astype(np.float32)
    cont_image = cv2.drawContours(im_crop,contours,-1,(0,255,0),1)

    #Preview
    #test3
    cv2. imshow("Test3",cont_image)
    cv2.imwrite(os.path.join(save_img_path,str(i)+'contview.jpg'),cont_image)
    k = cv2.waitKey(1000) & 0xff
    if k == 27:
        break


    cv2.imshow("predictions1", imga)
    cv2.imwrite(os.path.join(save_img_path,str(i)+'.jpg'),imga)
    k = cv2.waitKey(1000) & 0xff
    if k == 27:
        break
    i+=1

cv2.destroyAllWindows()




