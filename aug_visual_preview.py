from annotation_utils.coco.structs import COCO_Dataset
from logger import logger
from common_utils.image_utils import concat_n_images
from common_utils.cv_drawing_utils import cv_simple_image_viewer
from common_utils.file_utils import file_exists, make_dir_if_not_exists
import cv2
from common_utils.common_types.keypoint import Keypoint2D_List, Keypoint2D
# import printj
from imageaug import AugHandler, Augmenter as aug
from random import choice
from tqdm import tqdm
# import imgaug.augmenters as iaa


# PATH='/home/jitesh/3d/data/coco_data/mp_200_23_04_2020_15_37_00_coco-data'
# path = '/home/jitesh/3d/data/coco_data/sample_measure_coco_data'
# path = '/home/jitesh/3d/data/coco_data/measure_combined7'
# dest_folder_img_combined = f'{path}/img'
# dest_json_file_combined = f'{path}/json/measure-only.json'
path = '/home/pasonatech/labelme/ndds2coco/6_22/bolt_mark/type4'
dest_folder_img_combined = f'{path}'
dest_json_file_combined = f'{path}/HSR-coco.json'
dataset = COCO_Dataset.load_from_path(
    json_path=dest_json_file_combined,
    img_dir=dest_folder_img_combined
)
output = f'{path}/aug_vis'
make_dir_if_not_exists(output)
iaa = aug
# resize_save_path = 'test_resize.json'
handler_save_path = 'test_handler.json'
# if not file_exists(resize_save_path):
#     resize = aug.Resize(width=500, height=500)
#     resize.save_to_path(save_path=resize_save_path, overwrite=True)
#     logger.info(f'Created new Resize save.')
# else:
#     resize = aug.Resize.load_from_path(resize_save_path)
#     logger.info(f'Loaded Resize from save.')
# if not file_exists(handler_save_path):
#     handler = AugHandler(
#         [
#             # aug.Affine(scale = {"x": tuple([0.8, 1.2]), "y":tuple([0.8, 1.2])}, translate_percent= {"x": tuple([0.1, 0.11]), "y":tuple([0.1, 0.11])}, rotate= [-180, 180], order= [0, 0], cval= [0, 0], shear= [0,0]),
#             # aug.Crop(percent=[0.2, 0.5]),
#             # aug.Flipud(p=0.5),
#             # aug.Superpixels(),
#             # aug.Sharpen(alpha=[0,0.1], lightness=[0.8,1]),
#             # aug.Emboss(alpha=[0,0.1], strength=[0.8,1]),
#             # aug.AdditiveGaussianNoise(),
#             aug.Invert(p=1, per_channel=False),
#             # aug.Add(value=[-20,20], per_channel=True),
#             # aug.LinearContrast(alpha=[0.6,1.4], per_channel=True),
#             # aug.Grayscale(alpha=0.8),
#             # aug.Multiply(mul=[0.8,1.2], per_channel=False),
#             # aug.ElasticTransformation(alpha=[0,40], sigma=[4,6]),
#             # aug.PiecewiseAffine(scale=[0.0,0.05]),
#             aug.ContrastNormalization(alpha=[0.7,1], per_channel=True),
#             aug.AverageBlur(k=[1,7]),
#             aug.MotionBlur(k=[3,7], angle=[0,360]),
#             aug.BilateralBlur(d=[1,9]),
#             aug.EdgeDetect(alpha=[0,0.5]),
#             aug.DirectedEdgeDetect(alpha=[0,0.5], direction=[0,1]),
#             aug.Dropout(p=[0,0.15], per_channel=True),
#             aug.CoarseDropout(p=[0,0.5]),
#             # aug.Resize(),
#             aug.Grayscale(alpha=0.9, frequency=0.2),
#             # aug.BilateralBlur(d=[1,2])
#         ]
#     )
#     handler.save_to_path(save_path=handler_save_path, overwrite=True)
#     logger.info(f'Created new AugHandler save.')
# else:
    # handler = AugHandler.load_from_path(handler_save_path)
handler = AugHandler(aug_modes=
    [
        # aug.Affine(
        #     scale = {"x": tuple([0.8, 1]), "y":tuple([0.8, 1])},
        #     translate_percent= {"x": tuple([0.1, 0.11]), "y":tuple([0.1, 0.11])},
        #     rotate= [-180, 180],
        #     order= [0, 0],
        #     cval= [0, 0],
        #     shear= [0,0],
        #     frequency= 0.6).change_rotate_to_right_angle(),

        #aug.GaussianBlur(sigma=(0, 1.5),frequency=0.5),
        #aug.Crop(percent=[0, 0.05], frequency= 0.5),
        #aug.Flipud(p=0.5),
        #aug.Superpixels(),
        #aug.Sharpen(alpha=[0,0.5], lightness=[0.8,1], frequency= 0.5),
        #aug.Emboss(alpha=[0,0.5], strength=[0.8,1], frequency=0.5),
        #aug.AdditiveGaussianNoise(loc= 0, scale=[0,24.75], per_channel=1,frequency=0.8),
        #aug.Add(value=[-20,20], per_channel=True, frequency= 0.5),
        #aug.LinearContrast(alpha=[0.6,1.4], per_channel=True,frequency= 0.5),
        #aug.Dropout(p=[0,0.1], per_channel=True),
        #aug.AverageBlur(k=[1,2], frequency = 0.5),
        #aug.MotionBlur(k=[3,5], angle=[0,360], frequency = 0.5),


        # aug.Invert(p=0.5, per_channel=False,frequency=1),
        #aug.Grayscale(alpha=0.2),
        # aug.Multiply(mul=[0.8,1.2], per_channel=True),
        # aug.ElasticTransformation(alpha=[0,40], sigma=[4,6]),
        # aug.PiecewiseAffine(scale=[0.0,0.05]),
        #aug.ContrastNormalization(alpha=[0.7,1], per_channel=True),
        # iaa.BilateralBlur(d=(7, 7), sigma_color=(10, 250), sigma_space=(10, 250))

        #aug.Fliplr(p=0.5, frequency=0.7)
        # aug.BilateralBlur(d=[1,9]),
        #aug.EdgeDetect(alpha=[0,0.1]),
        # aug.DirectedEdgeDetect(alpha=[0,0.5], direction=[0,1]),
        # aug.CoarseDropout(p=[0,0.04], size_percent=0.05),
        # aug.Resize()
        
        ])
# handler = random_handler()
handler.save_to_path(save_path=handler_save_path, overwrite=True)
logger.info(f'Loaded AugHandler from save.')

img_buffer = []
aug_store = []
for i, coco_image in tqdm(enumerate(dataset.images)):
    img = cv2.imread(coco_image.coco_url)
    img_buffer.append(img)

    keypoints = []
    bbox = []
    segmentation = []
    ann_instance = 0

    for item in dataset.annotations:
        if item.image_id == coco_image.id:
            # keypoints_num = len(item.keypoints)
            ann_instance += 1

            # keypoints.append(item.keypoints)
            bbox.append(item.bbox)
            # segmentation.append(item.segmentation)
    
    # print(segmentation)
    image, bbox = handler(image=img, bounding_boxes=bbox)
    aug_store.append(image)
    # kpts_aug_list = keypoints[0].to_numpy(demarcation=True)[:, :2].reshape(ann_instance, keypoints_num, 2)
    # kpts_aug_list = [[[x, y, 2] for x, y in kpts_aug] for kpts_aug in kpts_aug_list]
    # keypoints = [Keypoint2D_List.from_list(kpts_aug, demarcation=True) for kpts_aug in kpts_aug_list]

    # print(image, keypoints, bbox, poly)
    
    cv2.imshow("aug", image)
    k = cv2.waitKey(50000) & 0xff
    if k == 27:
        break
    
   

cv2.destroyAllWindows()
