import os
import numpy as np
import json,random
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
import itertools
import cv2

path_of_raw_img = "/home/pasonatech/Desktop/trash_real_img"
# write a function that loads the dataset into detectron2's standard format
def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for _, v in imgs_anns.items():
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = list(itertools.chain.from_iterable(poly))

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

from detectron2.data import DatasetCatalog, MetadataCatalog
for d in ["train"]:
    DatasetCatalog.register("petbottle_" + d, lambda d=d: get_balloon_dicts("petbottle_" + d))
    MetadataCatalog.get("petbottle_" + d).set(thing_classes=["petbottle"])
petbottle_metadata = MetadataCatalog.get("petbottle")

#path od dataset
dataset_dicts = get_balloon_dicts(path_of_raw_img)
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=petbottle_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow("Frame",vis.get_image()[:, :, ::-1])
    cv2.waitKey(1000)

cv2.destroyAllWindows