"""
CenterNetV2CenterNetV2
Configurations and data loading code for CenterNetV2.

Reference from:
<Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla>
and
<pycocotools https://github.com/cocodataset/cocoapi>

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 Taco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 Taco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 Taco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 Taco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 Taco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
from scipy import ndimage
from skimage import transform

sys.path.append("..")
from config import Config
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from utils import utils as cocoutils
from utils.utils import resize_mask
import matplotlib.pyplot as plt
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import CenterNet2 config
sys.path.append(ROOT_DIR)  # To find local version of the library

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "centernetv2_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

DEFAULT_DATASET_YEAR = "2014"


############################################################
#  Configurations
############################################################


class TacoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "taco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if y
    # ou use a smaller GPU.
    IMAGES_PER_GPU = 4
    STEPS_PER_EPOCH = 9796
    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1
    PIC_NUM = 269

    # Number of classes (including background)
    NUM_CLASSES = 9  # COCO has 80 classes

    MODEL = 'train'
    GT_CHANNEL = 7
    L2_DECAY = 1e-4
    TRAIN_BN = True

    SCORE_THRESHOLD = 0.1

    VALIDATION_STEPS = 1


class CenterNetTestConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 3
    PIC_NUM = 269
    STEPS_PER_EPOCH = 200
    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 3  # COCO has 80 classes

    AUGMENTATION = None
    AUGMENT = False
    MODEL = 'test'
    FUSION = 'DLA'
    GT_CHANNEL = 7
    L2_DECAY = 1e-4
    TRAIN_BN = False

    SCORE_THRESHOLD = 0.1

    VALIDATION_STEPS = 1

############################################################
#  Dataset
############################################################


class TacoDataset(cocoutils.Dataset):
    def __init__(self, config, class_map=None):
        super().__init__(class_map=None)
        self.config = config
        self.num_classes = self.config.NUM_CLASSES


    def load_coco(self, config, dataset_dir, class_ids=None, class_nms=None, super_class_nms=None, return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        """

        coco = COCO("{}/annotations.json".format(dataset_dir))

        # Load all classes or a subset?
        if not class_ids and not class_nms and not super_class_nms:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates, some class_id are contented in same image.
            image_ids = list(set(image_ids))
        elif class_nms:
            class_ids = []
            image_ids = []
            for name in class_nms:
                group_class_ids = list(coco.getCatIds(catNms=[name]))
                class_ids.extend(group_class_ids)
                image_ids.extend(list(coco.getImgIds(catIds=group_class_ids)))
            # Remove duplicates, some class_id are contented in same image.
            image_ids = list(set(image_ids))
        elif super_class_nms:
            class_ids = []
            image_ids = []
            for name in super_class_nms:
                group_class_ids = list(coco.getCatIds(supNms=[name]))
                class_ids.extend(group_class_ids)
                image_ids.extend(list(coco.getImgIds(catIds=group_class_ids)))
            # Remove duplicates, some class_id are contented in same image.
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())
        print("sub_class_id: ", class_ids)
        self.sub_class_ids = class_ids
        # Add classes
        if class_ids:
            for i in class_ids:
                self.add_class("taco", i, coco.loadCats(i)[0]["supercategory"], coco.loadCats(i)[0]["name"])
        # elif class_nms:
        #     for name in class_nms:
        #         class_ids = list(coco.getCatIds(catNms=[name]))
        #         for i in class_ids:
        #             self.add_class("taco", i, coco.loadCats(i)[0]["supercategory"], coco.loadCats(i)[0]["name"])
        # elif super_class_nms:
        #     print("in super")
        #     for name in super_class_nms:
        #         class_ids = list(coco.getCatIds(supNms=[name]))
        #         print(class_ids)
        #         for i in class_ids:
        #             self.add_class("taco", i, coco.loadCats(i)[0]["supercategory"], coco.loadCats(i)[0]["name"])

        # Add images
        print("original image ids: ", image_ids)
        # self._image_ids = image_ids
        for i in image_ids:
            self.add_image(
                "taco", image_id=i,
                path=os.path.join(dataset_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def prepare_image(self, image_id):
        """use config to processing coco image size and others,
        augment: (deprecated. Use augmentation instead). If true, apply random
            image augmentation. Currently, only horizontal flipping is offered.
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
            For example, passing imgaug.augmenters.Fliplr(0.5) flips images
            right/left 50% of the time.

        Returns:
        image: [height, width, 3]
        image_meta: the original shape of the image and resizing and cropping.
        class_ids: [instance_count] Integer class IDs
        bbox: [instance_count, (y1, x1, y2, x2)]
        mask: [height, width, instance_count]. The height and width are those
            of the image.
        gt_y: [instance_count]
        gt_x: [instance_count]
        vector_mask: [height, width, 2*class_num]. Set pixel relative center vector.
        """
        # Load image and mask
        image = self.load_image(image_id=image_id)
        mask, class_ids = self.load_mask(image_id=image_id)
        original_shape = image.shape
        # print(original_shape)
        # print(type(original_shape))
        image, window, scale, padding, crop = cocoutils.resize_image(
            image,
            min_dim=self.config.IMAGE_MIN_DIM,
            min_scale=self.config.IMAGE_MIN_SCALE,
            max_dim=self.config.IMAGE_MAX_DIM,
            mode=self.config.IMAGE_RESIZE_MODE)
        mask = cocoutils.resize_mask(mask, scale, padding, 0, crop)
        _idx = np.sum(mask, axis=(0, 1)) > 16
        class_ids = class_ids[_idx]
        if len(class_ids) != 0:
            # print(class_ids)
            # [y, x, num_instance]
            mask = mask[:, :, _idx]
            # print(np.amax(mask, axis=(0, 1)))
            # Bounding boxes. Note that some boxes might be all zeros
            # if the corresponding mask got cropped out.
            # bbox: [num_instances, (y1, x1, y2, x2)]
            bbox = cocoutils.extract_bboxes(mask)
            gt_cy, gt_cx = cocoutils.gravity_center(mask)
            return image, class_ids, bbox, mask, gt_cy, gt_cx
        print("return nothing")
        return None

    def load_mask(self, image_id):
        """Load class_vector masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a float [height, width, instances_count].

        Returns:
        masks: A  array of shape [height, width, instances_count] with
            one mask per class.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "taco":
            return super(TacoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            if annotation['category_id'] not in self.sub_class_ids:
                continue
            class_id = self.map_source_class_id(
                "taco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? iscrowd == 1, used RLE format, If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            # [height, width, instances_count]
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            # [instances_count]
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(TacoDataset, self).load_mask(image_id)

    def image_info_to_tfrecords(self, dataset_dir, options=None):
        """Save Image_info to TFRecords document Override this to fit different requirement
        options = TFRecordCompressionType.ZLIB, TFRecordCompressionType.GZIP, TFRecordCompressionType.NONE
        """
        self.path = "{}/taco.tfrecord".format(dataset_dir)
        if os.path.exists(self.path):
            print("taco.tfrecord exists")
            return None
        writer = tf.io.TFRecordWriter(self.path, options)
        for image_id in range(self.num_images):
            results = self.prepare_image(image_id=image_id)
            if results is None:
                continue
            else:
                image, class_ids, bbox, mask, gt_cy, gt_cx = results
            gt_cy = gt_cy.tolist()
            gt_cx = gt_cx.tolist()
            image = image.tostring()
            class_ids = class_ids.tolist()
            mask = mask.astype(np.int8).flatten().tostring()
            feature_internal = {
                "class_ids": self.int64_feature(class_ids),
                "image_raw": self.bytes_feature(image),
                "masks_raw": self.bytes_feature(mask)
            }
            features_extern = tf.train.Features(feature=feature_internal)
            example = tf.train.Example(features=features_extern)
            example_str = example.SerializeToString()
            writer.write(example_str)
        writer.close()
        return None

    # The following two functions are from pycocotools with a few changes.
    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    def random_rotate_image(self, image):
        tf.compat.v1.set_random_seed(666)
        angle = np.random.uniform(low=-30.0, high=30.0)
        return ndimage.rotate(image, angle)
        # return transform(image.numpy(), angle).astype(np.int8)

    # 随机左右翻转图片
    def random_flip_lr(self, image):
        tf.compat.v1.set_random_seed(666)
        image_flip = tf.image.random_flip_left_right(image)
        return image_flip

    def random_flip_updown(self, image):
        tf.compat.v1.set_random_seed(666)
        image_flip = tf.image.random_flip_up_down(image)
        return image_flip

    # 随机变化图片亮度
    def random_brightness_image(self, image):
        tf.compat.v1.set_random_seed(666)
        image_bright = tf.image.random_brightness(image, max_delta=0.3)
        return image_bright

    # 随机裁剪图片
    def random_crop_image(self, image):
        tf.compat.v1.set_random_seed(666)
        # 裁剪后图片分辨率保持160x160,3通道
        image_crop = tf.image.random_crop(image, [self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM, 3])
        return image_crop

    def gravity_center(self, mask):
        """Compute gravity center of each mask
        :param mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
        :return: [gt_cy, gt_cx] : num_instances, num_instances
        """
        gt = []
        _idx = []
        for i in range(np.shape(mask)[-1]):
            m = mask[:, :, i]
            if np.sum(m, axis=(0, 1)) > 25:
                res, thresh = cv2.threshold(m, 0.5, 255, 0)
                moments = cv2.moments(thresh)
                # print("m00: ", moments["m00"], moments["m10"], moments["m01"])
                cx = (moments["m10"] / moments["m00"])
                cy = (moments["m01"] / moments["m00"])

                horizontal_indicies = np.where(np.any(m, axis=0))[0]
                # vertical for y axis
                vertical_indicies = np.where(np.any(m, axis=1))[0]
                if horizontal_indicies.shape[0]:
                    x1, x2 = horizontal_indicies[[0, -1]]
                    y1, y2 = vertical_indicies[[0, -1]]
                    # x2 and y2 should not be part of the box. Increment by 1.
                    x2 += 1
                    y2 += 1
                else:
                    # No mask for this instance. Might happen due to
                    # resizing or cropping. Set bbox to zeros
                    x1, x2, y1, y2 = 0, 0, 0, 0
                gt_exid = [cy, cx, y1, x1, y2, x2]
                gt.append(gt_exid)
                _idx.append(i)
        return gt, _idx

    def generator(self, batch_size):
        """Load and return ground truth data for an image (image, mask, bounding boxes).
        Returns:
        image: [height, width, 3]
        # shape: the original shape of the image before resizing and cropping.
        gt: {   class_ids: [instance_count] Integer class IDs,
                bbox: [instance_count, (y1, x1, y2, x2)]
            }
        stride_mask: [height, width, class_num*2]. The height and width are 1/4 those
            of the image.
        """
        print("-----------------------")
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        mean = tf.reshape(mean, [1, 1, 3])
        std = tf.reshape(std, [1, 1, 3])
        reader = tf.data.TFRecordDataset(self.path)
        feature_description = {
                               'class_ids': tf.io.VarLenFeature(tf.int64),
                               'image_raw': tf.io.FixedLenFeature([], tf.string),
                               'masks_raw': tf.io.FixedLenFeature([], tf.string)}

        def _parse_example(example):
            return tf.io.parse_single_example(example, feature_description)

        dataset = reader.map(_parse_example)
        rs = tf.compat.v1.set_random_seed(666)
        b = 0
        imgs = []
        gt_batch = []
        masks_batch = []
        while True:
            for data in dataset:
                class_ids = tf.sparse.to_dense(data['class_ids'])
                num_g = tf.shape(class_ids)[0]
                img = tf.io.decode_raw(data['image_raw'], out_type=tf.uint8)
                img = tf.reshape(img, (self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM, 3))
                masks = tf.io.decode_raw(data['masks_raw'], out_type=tf.uint8)
                masks = tf.reshape(masks, (self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM, num_g))
                pic = tf.concat([img, masks], axis=2)
                pic = tf.image.random_flip_left_right(pic)
                pic = tf.image.random_flip_up_down(pic)
                size = tf.cast(self.config.IMAGE_MAX_DIM * 3 / 4, tf.int32)
                pic = tf.image.random_crop(pic, [size, size, num_g + 3])
                pic = self.random_rotate_image(pic)
                # for i in range(30):
                #     ret = tf.cast(tf.random.normal([], mean=312, stddev=39, dtype=tf.float32, seed=rs), tf.int32)
                #     print(ret)
                ret = tf.cast(tf.random.normal([], mean=312, stddev=78, dtype=tf.float32, seed=rs), tf.int32)
                ret = tf.cond(ret > 351, lambda: 351, lambda: ret)
                w = tf.cast(tf.cond(ret < 273, lambda: 273, lambda: ret), tf.int32)
                map = tf.cast(self.config.IMAGE_MAX_DIM / 4, tf.int32)
                pic = tf.image.resize(pic, [w, w])
                pic = tf.image.resize_with_crop_or_pad(pic, self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM)

                img = pic[..., 0:3]
                img = tf.image.random_brightness(img, max_delta=0.3)
                img = (tf.cast(img, tf.float32) / 255. - mean) / std
                masks = pic[..., 3:]
                gt_exid, idx = self.gravity_center(masks)
                if len(idx) is 0:
                    print("Image has no Mask...")
                    continue
                # [y, x, n]
                masks = tf.gather(masks, idx, axis=2)
                # print(tf.shape(masks))
                class_ids = tf.expand_dims(tf.cast(tf.gather(class_ids, idx, axis=0), tf.float32), -1) + 1
                gt_exid = tf.cast(gt_exid, tf.float32)
                gt = tf.concat([gt_exid, class_ids], axis=-1)
                masks = tf.image.resize(masks, [map, map])
                masks = tf.cast(tf.cast((masks + 0.6), tf.uint8), tf.float32)
                # plt.imshow(img)
                # plt.show()
                # mm = tf.expand_dims(masks[:, :, 0], -1)
                # masks = tf.concat([mm, mm, mm], axis=-1)
                # plt.imshow(masks)
                # plt.show()
                instance_num = tf.shape(gt)[0]
                gt = tf.cond(
                    instance_num > self.config.MAX_GT_INSTANCES,
                    lambda: gt[0:self.config.MAX_GT_INSTANCES, ...],
                    lambda: tf.pad(gt, [[0,self.config.MAX_GT_INSTANCES - instance_num], [0, 0]], mode='CONSTANT')
                )
                masks = tf.cond(
                    instance_num > self.config.MAX_GT_INSTANCES,
                    lambda: masks[..., 0:self.config.MAX_GT_INSTANCES],
                    lambda: tf.pad(masks, [[0, 0], [0, 0], [0, self.config.MAX_GT_INSTANCES - instance_num]], mode='CONSTANT')
                )
                imgs.append(img)
                gt_batch.append(gt)
                masks_batch.append(masks)
                b += 1
                if b >= batch_size:
                    inputs = (imgs, gt_batch, masks_batch)
                    targets = []
                    yield inputs, targets
                    imgs = []
                    gt_batch = []
                    masks_batch = []
                    b = 0

    def random_flip(self, image):
        # axis 0 垂直翻转，1水平翻转 ，-1水平垂直翻转，2不翻转，各自以25%的可能性
        axis = np.random.randint(low=-1, high=3)
        if axis != 2:
            image = cv2.flip(image, axis)
        return image


    def generator_from_raw(self, batch_size):
        """Load and return ground truth data for an image (image, mask, bounding boxes).
        Returns:
        image: [height, width, 3]
        # shape: the original shape of the image before resizing and cropping.
        gt: {   class_ids: [instance_count] Integer class IDs,
                bbox: [instance_count, (y1, x1, y2, x2)]
            }
        stride_mask: [height, width, class_num*2]. The height and width are 1/4 those
            of the image.
        """
        print("-----------------------")
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        mean = np.reshape(mean, [1, 1, 3])
        std = np.reshape(std, [1, 1, 3])

        b = 0
        imgs = []
        gt_batch = []
        masks_batch = []
        while True:
            for image_id in range(self.num_images):
                results = self.prepare_image(image_id=image_id)
                if results is None:
                    continue
                else:
                    image, class_ids, bbox, masks, gt_cy, gt_cx = results

                pic = np.concatenate([image, masks], axis=2)
                pic = self.random_flip(pic)
                size = int(self.config.IMAGE_MAX_DIM * 3 / 4)
                x_offset = np.random.randint(low=0, high=self.config.IMAGE_MAX_DIM/4, size=1)[0]
                y_offset = np.random.randint(low=0, high=self.config.IMAGE_MAX_DIM/4, size=1)[0]
                pic = pic[x_offset:x_offset + size, y_offset:y_offset + size, :]

                pic = self.random_rotate_image(pic)
                ret = (np.random.uniform(280, 390, size=1) / 2).astype(np.int32) * 2
                map = int(self.config.IMAGE_MAX_DIM / 4)
                pic = cv2.resize(pic, (ret, ret))
                pad_size = int((self.config.IMAGE_MAX_DIM - ret) / 2)
                pic = np.pad(pic, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'constant')
                img = pic[..., 0:3]
                # slope = np.random.uniform(0.8, 1.2, 1)
                # # # 亮度调整系数
                # bias = np.random.uniform(-0.2, 0.2, 1)
                # # # 图像亮度和对比度调整
                # img = img * slope + bias
                # # # 灰度值截断，防止超出255
                # img = np.clip(img, 0, 255)
                # #
                # img = img.astype(np.uint8)
                # img = tf.image.random_brightness(img, max_delta=0.3)
                img = (img / 255. - mean) / std
                masks = pic[..., 3:]
                gt_exid, idx = self.gravity_center(masks)
                if len(idx) is 0:
                    continue
                # [y, x, n]
                masks = masks[..., idx]
                class_ids = np.expand_dims(class_ids[idx], axis=-1)
                gt = np.concatenate([gt_exid, class_ids], axis=-1)
                masks = cv2.resize(masks, (map, map), interpolation=cv2.INTER_LINEAR)
                # masks = tf.cast(tf.cast((masks + 0.6), np.int32), np.float32)
                # plt.imshow(img)
                # plt.show()
                # print(np.shape(masks))
                if len(np.shape(masks)) is 2:
                    masks = np.expand_dims(masks, -1)
                # plt.imshow(masks[:, :, 0])
                # plt.show()
                instance_num = np.shape(gt)[0]
                if instance_num <= self.config.MAX_GT_INSTANCES:
                    gt = np.pad(gt, ((0, self.config.MAX_GT_INSTANCES - instance_num), (0, 0)), mode='constant')
                else:
                    gt = gt[:self.config.MAX_GT_INSTANCES, :]
                if instance_num <= self.config.MAX_GT_INSTANCES:
                    masks = np.pad(masks, ((0, 0), (0, 0), (0, self.config.MAX_GT_INSTANCES - instance_num)), mode='constant')
                else:
                    masks = masks[:, :, 0:self.config.MAX_GT_INSTANCES]
                imgs.append(img)
                gt_batch.append(gt)
                masks_batch.append(masks)
                b += 1
                if b >= batch_size:
                    inputs = (imgs, gt_batch, masks_batch)
                    targets = []
                    yield inputs, targets
                    imgs = []
                    gt_batch = []
                    masks_batch = []
                    b = 0

    def for_sequence(self, image_id):
        """Load and return ground truth data for an image (image, mask, bounding boxes).
        Returns:
        image: [height, width, 3]
        # shape: the original shape of the image before resizing and cropping.
        gt: {   class_ids: [instance_count] Integer class IDs,
                bbox: [instance_count, (y1, x1, y2, x2)]
            }
        stride_mask: [height, width, class_num*2]. The height and width are 1/4 those
            of the image.
        """
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        mean = np.reshape(mean, [1, 1, 3])
        std = np.reshape(std, [1, 1, 3])

        results = self.prepare_image(image_id=image_id)
        if results is None:
            return None
        else:
            image, class_ids, bbox, masks, gt_cy, gt_cx = results

        pic = np.concatenate([image, masks], axis=2)
        pic = self.random_flip(pic)
        size = int(self.config.IMAGE_MAX_DIM * 3 / 4)
        x_offset = np.random.randint(low=0, high=self.config.IMAGE_MAX_DIM/4, size=1)[0]
        y_offset = np.random.randint(low=0, high=self.config.IMAGE_MAX_DIM/4, size=1)[0]
        pic = pic[x_offset:x_offset + size, y_offset:y_offset + size, :]

        pic = self.random_rotate_image(pic)
        ret = (np.random.uniform(280, 390, size=1) / 2).astype(np.int32) * 2
        map = int(self.config.IMAGE_MAX_DIM / 4)
        pic = cv2.resize(pic, (ret, ret))
        pad_size = int((self.config.IMAGE_MAX_DIM - ret) / 2)
        pic = np.pad(pic, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'constant')
        img = pic[..., 0:3]
        img = (img / 255. - mean) / std
        masks = pic[..., 3:]
        gt_exid, idx = self.gravity_center(masks)
        if len(idx) is 0:
            return None
        # [y, x, n]
        masks = masks[..., idx]
        class_ids = np.expand_dims(class_ids[idx], axis=-1)
        gt = np.concatenate([gt_exid, class_ids], axis=-1)
        masks = cv2.resize(masks, (map, map), interpolation=cv2.INTER_LINEAR)
        # masks = tf.cast(tf.cast((masks + 0.6), np.int32), np.float32)
        # plt.imshow(img)
        # plt.show()
        # print(np.shape(masks))
        if len(np.shape(masks)) is 2:
            masks = np.expand_dims(masks, -1)
        # plt.imshow(masks[:, :, 0])
        # plt.show()
        instance_num = np.shape(gt)[0]
        if instance_num <= self.config.MAX_GT_INSTANCES:
            gt = np.pad(gt, ((0, self.config.MAX_GT_INSTANCES - instance_num), (0, 0)), mode='constant')
        else:
            gt = gt[:self.config.MAX_GT_INSTANCES, :]
        if instance_num <= self.config.MAX_GT_INSTANCES:
            masks = np.pad(masks, ((0, 0), (0, 0), (0, self.config.MAX_GT_INSTANCES - instance_num)), mode='constant')
        else:
            masks = masks[:, :, 0:self.config.MAX_GT_INSTANCES]
        return img, gt, masks

    def for_val_sequence(self, image_id):
        """Load and return ground truth data for an image (image, mask, bounding boxes).
        Returns:
        image: [height, width, 3]
        # shape: the original shape of the image before resizing and cropping.
        gt: {   class_ids: [instance_count] Integer class IDs,
                bbox: [instance_count, (y1, x1, y2, x2)]
            }
        stride_mask: [height, width, class_num*2]. The height and width are 1/4 those
            of the image.
        """
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        mean = np.reshape(mean, [1, 1, 3])
        std = np.reshape(std, [1, 1, 3])

        results = self.prepare_image(image_id=image_id)
        if results is None:
            return None
        else:
            image, class_ids, bbox, masks, gt_cy, gt_cx = results
        img = (image / 255. - mean) / std

        map = int(self.config.IMAGE_MAX_DIM / 4)
        masks = masks.astype(np.float32)
        if len(np.shape(masks)) is 2:
            masks = np.expand_dims(masks, axis=-1)
        masks = cv2.resize(masks, (map, map), interpolation=cv2.INTER_LINEAR)
        class_ids = np.expand_dims(class_ids, axis=-1)
        gt_cy = np.expand_dims(gt_cy, axis=-1)
        gt_cx = np.expand_dims(gt_cx, axis=-1)
        gt = np.concatenate([gt_cy, gt_cx, bbox, class_ids], axis=-1)
        instance_num = np.shape(gt)[0]
        if instance_num <= self.config.MAX_GT_INSTANCES:
            gt = np.pad(gt, ((0, self.config.MAX_GT_INSTANCES - instance_num), (0, 0)), mode='constant')
        else:
            gt = gt[:self.config.MAX_GT_INSTANCES, :]
        if instance_num <= self.config.MAX_GT_INSTANCES:
            masks = np.pad(masks, ((0, 0), (0, 0), (0, self.config.MAX_GT_INSTANCES - instance_num)), mode='constant')
        else:
            masks = masks[:, :, 0:self.config.MAX_GT_INSTANCES]
        return img, gt, masks


class SequenceData(tf.keras.utils.Sequence):
    def __init__(self, num_imgs, batch_size, dataset, img_idx):
        # 初始化所需的参数
        self.batch_size = batch_size
        self.dataset = dataset
        self.num_imgs = num_imgs
        self.img_idx = img_idx

    def __len__(self):
        return np.math.ceil(self.num_imgs / self.batch_size)

    def __getitem__(self, idx):
        # 迭代器部分
        imgs = []
        gt_batch = []
        masks_batch = []
        b = 0
        id = idx * self.batch_size
        while True:
            if self.img_idx >= self.num_imgs:
                self.img_idx = 0
            if id >= self.num_imgs:
                id = 0
            if b >= self.batch_size:
                inputs = (imgs, gt_batch, masks_batch)
                targets = []
                return inputs, targets
            res = self.dataset.for_sequence(image_id=id)
            if res is not None:
                img, gt, masks = res
                imgs.append(img)
                gt_batch.append(gt)
                masks_batch.append(masks)
                b += 1
            id += 1
            self.img_idx += 1

class SequenceVal(tf.keras.utils.Sequence):
    def __init__(self, num_imgs, batch_size, dataset, img_idx):
        # 初始化所需的参数
        self.batch_size = batch_size
        self.dataset = dataset
        self.num_imgs = num_imgs
        self.img_idx = img_idx

    def __len__(self):
        return np.math.ceil(self.num_imgs / self.batch_size)

    def __getitem__(self, idx):
        # 迭代器部分
        imgs = []
        gt_batch = []
        masks_batch = []
        b = 0
        id = idx * self.batch_size
        while True:
            if self.img_idx >= self.num_imgs:
                self.img_idx = 0
            if id >= self.num_imgs:
                id = 0
            if b >= self.batch_size:
                inputs = (imgs, gt_batch, masks_batch)
                targets = []
                return inputs, targets
            res = self.dataset.for_val_sequence(image_id=id)
            if res is not None:
                img, gt, masks = res
                imgs.append(img)
                gt_batch.append(gt)
                masks_batch.append(masks)
                b += 1
            id += 1
            self.img_idx += 1