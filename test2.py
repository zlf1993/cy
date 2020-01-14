from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils import tfrecord_voc_utils as voc_utils
import numpy as np
import CenterNet as net
import os
from coco.coco import CenterNetTestConfig, CocoDataset
from utils import utils as cocoutils
from PIL import Image
from utils import visualize

# Set Coco
ROOT_DIR = os.path.abspath("../")

Config = CenterNetTestConfig()
Config.display()

dataset = CocoDataset()
COCO_DIR = ROOT_DIR + "/coco2014"
dataset.load_coco(Config, COCO_DIR, "train", class_ids=[1, 17, 18])
dataset.prepare()

centernet = net.CenterNet(Config, "Creature")

# centernet.train_epochs(dataset, valset, Config, 50)
#image = Image.open('COCO_val2014_000000000761.jpg')
image = Image.open('COCO_val2014_000000000241.jpg')
# image = Image.open('COCO_train2014_000000000113.jpg')
#image = Image.open('COCO_train2014_000000005083.jpg')

image = np.array(image)
image, window, scale, padding, crop = cocoutils.resize_image(
            image,
            min_dim=Config.IMAGE_MIN_DIM,
            min_scale=Config.IMAGE_MIN_SCALE,
            max_dim=Config.IMAGE_MAX_DIM,
            mode=Config.IMAGE_RESIZE_MODE)

print(np.shape(image))
# select_scores, select_center, select_bbox, select_class_id, fcn, objvector
[scores, center, bbox, class_id, fcn, objvector] = centernet.test_one_image(image)
if np.shape(scores)[0] > 50:
    num_select = 50
else:
    num_select = np.shape(scores)[0]
select_scores = scores[0, 0:20, ...]
select_center = center[0, 0:20, ...]
select_class_id = class_id[0, 0:20, ...]
select_bbox = bbox[0, 0:20, ...]
print(select_center - select_bbox[...,[0,1]])

# display_instances(image, centers, boxes, masks, class_ids, class_names,
#                       scores=None, title="",
#                       figsize=(16, 16), ax=None,
#                       show_mask=False, show_bbox=True, show_centers=True,
#                       colors=None, captions=None):
visualize.display_instances(image, select_center, select_bbox, None, select_class_id,  dataset.class_names, show_bbox=True)

# print(select_scores)
# print(select_center)
# print(select_bbox)
# print(select_class_id)



