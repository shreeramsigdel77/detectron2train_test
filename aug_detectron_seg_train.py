# save dataset configuration to config file

from annotation_utils.coco.structs import COCO_Dataset
from annotation_utils.dataset.config.dataset_config import \
    DatasetConfigCollectionHandler, DatasetConfigCollection, DatasetConfig
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from imgaug import augmenters as iaa
import imgaug as ia
from imgaug.augmentables.polys import PolygonsOnImage
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.bbs import BoundingBoxesOnImage
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data import build_detection_train_loader
from detectron2.structures import BoxMode
from logger import logger
from common_utils.cv_drawing_utils import cv_simple_image_viewer, draw_bbox, draw_keypoints, draw_segmentation
from common_utils.common_types.keypoint import Keypoint2D_List, Keypoint2D
from common_utils.common_types.bbox import BBox
from common_utils.common_types.segmentation import Segmentation, Polygon
from common_utils.file_utils import file_exists
from imageaug import AugHandler, Augmenter as aug, AugVisualizer

import cv2, os
import numpy as np
import copy
import torch




class Test_Keypoint_Trainer(DefaultTrainer):
    def __init__(self, cfg, mapper):
        super().__init__(cfg=cfg)
        self.mapper = mapper

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg=cfg, mapper=mapper)

def load_augmentation_settings(handler_save_path: str):

    if not file_exists(handler_save_path):
        handler = AugHandler(
            [
        aug.Affine(
            scale = {"x": tuple([0.8, 1]), "y":tuple([0.8, 1])},
            translate_percent= {"x": tuple([0.1, 0.11]), "y":tuple([0.1, 0.11])},
            rotate= [-180, 180],
            order= [0, 0],
            cval= [0, 0],
            shear= [0,0],
            frequency= 0.6).change_rotate_to_right_angle(),

        aug.GaussianBlur(sigma=(0, 1.5),frequency=0.5),
        #aug.Crop(percent=[0, 0.1], frequency= 0.5),
        #aug.Flipud(p=0.5),
        aug.Sharpen(alpha=[0,0.5], lightness=[0.8,1], frequency= 0.5),
        aug.Emboss(alpha=[0,0.5], strength=[0.8,1], frequency=0.5),
        aug.AdditiveGaussianNoise(loc= 0, scale=[0,24.75], per_channel=1,frequency=0.8),
        aug.Add(value=[-20,20], per_channel=True, frequency= 0.5),
        aug.LinearContrast(alpha=[0.6,1.4], per_channel=True,frequency= 0.5),
        aug.Dropout(p=[0,0.01], per_channel=True),
        aug.AverageBlur(k=[1,2], frequency = 0.5),
        aug.MotionBlur(k=[3,5], angle=[0,360], frequency = 0.5),
        
        aug.Fliplr(p=0.5, frequency=0.7)
            ]
        )
        handler.save_to_path(save_path=handler_save_path, overwrite=True)
    else:
        handler = AugHandler.load_from_path(handler_save_path)

    return handler

def make_config_file_from_datasets(datasets_dict: {}, config_path: str):
    handler = DatasetConfigCollectionHandler()

    dataset_list = []

    for key in datasets_dict:
        dataset_list.append(
             DatasetConfig(
                        img_dir=key,
                        ann_path=datasets_dict[key],
                        ann_format='coco'
            )
        )
    handler.append(
        DatasetConfigCollection(
            dataset_list
        )
    )

    handler.save_to_path(config_path, overwrite=True)


def combine_dataset_from_config_file(config_path: str, dest_folder_img: str, dest_json_file: str ):
    # combine dataset
    combined_dataset = COCO_Dataset.combine_from_config(
        config_path=config_path,
        img_sort_attr_name='file_name',
        show_pbar=True
    )
    combined_dataset.move_images(
        dst_img_dir=dest_folder_img,
        preserve_filenames=False,
        update_img_paths=True,
        overwrite=True,
        show_pbar=True
    )
    combined_dataset.save_to_path(save_path=dest_json_file, overwrite=True)

def register_dataset_to_detectron(instance_name: str,img_dir_path: str, ann_path: str):
    register_coco_instances(
        name=instance_name,
        metadata={},
        json_file=ann_path,
        image_root=img_dir_path
    )
    MetadataCatalog.get(instance_name).thing_classes = ['mark']

    
def setup_config_file(instance_name:str, cfg):

    model_config_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"    
    #model_config_path = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(model_config_path))
    cfg.DATASETS.TRAIN = (instance_name,)
    cfg.DATASETS.TEST = (instance_name,)  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config_path)  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = (
        5000
    )  # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        512
    )  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 2 classes
    #print(cfg.INPUT.MIN_SIZE_TRAIN)
    #Size of the smallest side of the image during training
    cfg.INPUT.MIN_SIZE_TRAIN = (1024,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1024
    # Sample size of smallest side by choice or random selection from range give by
    # INPUT.MIN_SIZE_TRAIN
    #cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    # Maximum size of the side of the image during training
    print(cfg.INPUT.MIN_SIZE_TRAIN)
    print(cfg.INPUT.MAX_SIZE_TRAIN)
    

    return cfg

class Counter:
    def __init__(self):
        self.count = 0
    
    def count_add(self):
        self.count += 1
count = Counter()

aug_save_path = "/home/pasonatech/detectron/detectron2/gbox/aug_vis/"

def mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    #print(dataset_dict["file_name"])
    #cv2.imwrite(os.path.join(aug_save_path, f'{str(count.count)}_asd.jpg'),image)

    handler = load_augmentation_settings(handler_save_path = 'test_handler.json')
    for i in range(len(dataset_dict["annotations"])):
        dataset_dict["annotations"][i]["segmentation"] = []
    image, dataset_dict = handler(image=image, dataset_dict_detectron=dataset_dict)
    
    annots = []

    
    for item in dataset_dict["annotations"]:
        annots.append(item)

    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    instances = utils.annotations_to_instances(annots, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    if True:
        vis_img = image.copy()
        bbox_list = [BBox.from_list(vals) for vals in dataset_dict["instances"].gt_boxes.tensor.numpy().tolist()]
        # seg_list = [Segmentation([Polygon.from_list(poly.tolist(), demarcation=False) for poly in seg_polys]) for seg_polys in dataset_dict["instances"].gt_masks.polygons]
        for bbox in (bbox_list):
            # if len(seg) > 0 and False:
            #     vis_img = draw_segmentation(img=vis_img, segmentation=seg, transparent=True)
            vis_img = draw_bbox(img=vis_img, bbox=bbox)
        
        height,width,channels = vis_img.shape
        print(height,width)
        if count.count<100:
            cv2.imwrite(os.path.join(aug_save_path, f'{str(count.count)}.jpg'),vis_img)
        count.count_add()
        aug_vis.step(vis_img)


    return dataset_dict




if __name__ == "__main__":

    instance_name = "marker"
    # config_path = "scratch_config.yaml"
    #dest_folder_img_combined = "/home/pasonatech/Desktop/7/7_15/coco-data/combine" #random bolt color
    #dest_folder_img_combined = "/home/pasonatech/Desktop/7/7_27/combine-coco"  # silverboltdist

    dest_folder_img_combined = "/home/pasonatech/UEoutputs/7/coco/bolt_markMap_2020.07.22-10.55.53/marker"

    dest_json_file_combined = dest_folder_img_combined+"/HSR-coco.json"

    # instance_name = "hsr"
    # dest_folder_img_combined = "./dummy_data/18_03_2020_18_03_10_coco-data"
    # dest_json_file_combined = "./dummy_data/18_03_2020_18_03_10_coco-data/HSR-coco.json"
    # datasets_dict_img_annot = {
    #     './dummy_data/18_03_2020_18_03_10_coco-data' : './dummy_data/18_03_2020_18_03_10_coco-data/HSR-coco.json',
    #     './dummy_data/31_03_2020_17_37_04_coco-data': './dummy_data/31_03_2020_17_37_04_coco-data/HSR-coco.json'
    # }

    # make_config_file_from_datasets(datasets_dict= datasets_dict_img_annot, config_path= config_path)
    # combine_dataset_from_config_file(config_path= config_path, dest_folder_img= "./combined_img", dest_json_file= "./combined.json" )
    register_dataset_to_detectron(instance_name=instance_name,img_dir_path= dest_folder_img_combined, ann_path = dest_json_file_combined)

    cfg = setup_config_file(instance_name=instance_name, cfg=get_cfg())
    
    aug_vis = AugVisualizer(
        vis_save_path='aug_vis.png',
        wait=None
    )

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # TODO: add mapper
    trainer = Test_Keypoint_Trainer(cfg,mapper)
    trainer.resume_or_load(resume=True)
    trainer.train()



