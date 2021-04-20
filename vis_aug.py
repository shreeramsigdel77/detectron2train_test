from annotation_utils.coco.structs import COCO_Dataset
from logger import logger
from common_utils.image_utils import concat_n_images
from common_utils.cv_drawing_utils import cv_simple_image_viewer
from common_utils.file_utils import file_exists
import cv2,os
from common_utils.common_types.keypoint import Keypoint2D_List, Keypoint2D
from common_utils.common_types.segmentation import Segmentation, Polygon
import shapely
from shapely import ops
import numpy as np
from common_utils.cv_drawing_utils import draw_keypoints, cv_simple_image_viewer, draw_bbox, draw_segmentation

from imageaug import AugHandler, Augmenter as aug

save_img_path = "/home/pasonatech/detectron/detectron2/gbox/vis_image/"

dataset = COCO_Dataset.load_from_path(
    
    json_path='/home/pasonatech/combined_cocooutput/HSR-coco.json',
    img_dir='/home/pasonatech/combined_cocooutput'

    #json_path='/home/pasonatech/aug_real_combine/aug_sim_com_garbage/HSR-coco.json',
    #img_dir='/home/pasonatech/aug_real_combine/aug_sim_com_garbage'
)

resize_save_path = 'test_resize.json'
handler_save_path = 'test_handler.json'
if not file_exists(handler_save_path):
    handler = AugHandler(
        [
            aug.Crop(percent=[0.2, 0.5]),
            aug.Flipud(p=0.5),
            aug.Superpixels()
            # aug.Sharpen(alpha=[-1,0.1], lightness=[0,3])
        ]
    )
    handler.save_to_path(save_path=handler_save_path, overwrite=True)
    logger.info(f'Created new AugHandler save.')
else:
    handler = AugHandler.load_from_path(handler_save_path)
    logger.info(f'Loaded AugHandler from save.')
i=0
img_buffer = []
for coco_image in dataset.images:
    img = cv2.imread(coco_image.coco_url)
    img_buffer.append(img)

    keypoints = []
    bbox = []
    segmentation = []
    ann_instance = 0
    
    for item in dataset.annotations:
        if item.image_id == coco_image.id:
            keypoints_num = len(item.keypoints)
            ann_instance += 1

            if len(item.keypoints) != 0:
                keypoints.append(item.keypoints)
            bbox.append(item.bbox)
            
            if item.segmentation:
                if len(item.segmentation) != 0:

                    for segmentation_list in item.segmentation:
                        if len(segmentation_list.to_list()) < 5:
                            logger.red(coco_image.coco_url)
                            logger.red(item.segmentation)
                            logger.red(f"segmentation has abnormal point count {segmentation_list}" )
                            continue 
                            
                        poly_shapely = segmentation_list.to_shapely()
                        poly_list = list(zip(*poly_shapely.exterior.coords.xy))
                        
                        line_non_simple = shapely.geometry.LineString(poly_list)
                        mls = shapely.ops.unary_union(line_non_simple)
                        polygons = list(shapely.ops.polygonize(mls))
                        poly_shapely = polygons

                        segmentation.append(Segmentation.from_shapely(poly_shapely))
    image, bbox, poly = handler(image=img, bounding_boxes=bbox, polygons=segmentation)
    poly = [Segmentation(polygon_list=[pol]) for pol in poly]
    print("total bbox after")
    
    for polygons in poly:
        image = draw_segmentation(img=image, segmentation=polygons, transparent=True)
    
    
    print("finished drawing handler")
    cv2.imshow("Window", image)
    cv2.waitKey(4000)
    cv2.imwrite(os.path.join(save_img_path,str(i)+'.jpg'),image)
    i+=1
