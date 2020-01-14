from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from mobilenet_v3_block import BottleNeck, h_swish
from yolov3_layer_utils import upsample_layer, yolo_conv2d, yolo_block
import numpy as np
import sys
import os
sys.path.append("../")
from utils.utils import resize_image
from utils.visualize import display_instances


def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("", ""))
        text += "  {}".format(array.dtype)
    print(text)


class EpochRecord(tf.keras.callbacks.Callback):
    def __init__(self, name):
        super(EpochRecord, self).__init__()
        self.name = name

    def on_epoch_end(self, epoch, logs={}):
        if not os.path.exists(self.name+"/epoch.txt"):
            file = open(self.name+"/epoch.txt", 'w')
            file.write("0")
            file.close()
        file = open(self.name+"/epoch.txt", 'r')
        epoch = int(str(file.readline()))
        file.close()
        epoch += 1
        epoch = str(epoch)
        file = open(self.name + "/epoch.txt", 'w')
        file.write(epoch)
        file.close()


class CentLoss(tf.keras.layers.Layer):
    def __init__(self, batch_size, num_class, decay, **kwargs):
        super(CentLoss, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.num_class = num_class
        self.decay = decay

    def call(self, inputs, **kwargs):
        keypoints, preg, ground_truth = inputs
        losses = self._centernet_loss(ground_truth, keypoints, preg)
        total_loss = tf.reduce_sum(losses*tf.convert_to_tensor(self.decay))
        self.add_loss(total_loss, inputs=True)
        self.add_metric(losses[0], aggregation="mean", name="keypoint_loss")
        self.add_metric(losses[1], aggregation="mean", name="size_loss")
        return total_loss

    def _centernet_loss(self, ground_truth, keypoints, preg):
        pshape = [tf.shape(keypoints)[1], tf.shape(keypoints)[2]]
        h = tf.range(0., tf.cast(pshape[0], tf.float32), dtype=tf.float32)
        w = tf.range(0., tf.cast(pshape[1], tf.float32), dtype=tf.float32)

        # shape of coordinate equals [h_y_num, w_x_mun]
        [meshgrid_x, meshgrid_y] = tf.meshgrid(w, h)
        total_loss = []
        for i in range(self.batch_size):
            # loss = [keypoints_loss, size_loss, offset_loss, fcn_loss, vector_loss]
            loss = self._compute_one_image_loss(keypoints[i, ...], preg[i, ...], ground_truth[i, ...],
                                                meshgrid_y, meshgrid_x, 8.0, pshape)
            total_loss.append(loss)
        mean_loss = tf.reduce_mean(total_loss, axis=0)
        return mean_loss

    def _compute_one_image_loss(self, keypoints, preg, ground_truth, meshgrid_y,
                                meshgrid_x, stride, pshape):
        # find valid num_gt, ignore padding parts
        slice_index = tf.argmin(ground_truth, axis=0)[0]
        ground_truth = tf.gather(ground_truth, tf.range(0, slice_index, dtype=tf.int64))
        # gt_basic = [gt_y, gt_x, gt_top, gt_left, gt_bot, gt_right, class_ids]
        ngbbox_y = ground_truth[..., 0] / stride
        ngbbox_x = ground_truth[..., 1] / stride
        ngbbox_h = (ground_truth[..., 2] + ground_truth[..., 4]) / stride
        ngbbox_w = (ground_truth[..., 3] + ground_truth[..., 5]) / stride
        class_id = tf.cast(ground_truth[..., 6], dtype=tf.int32)

        # object_vector_mask shape: [y, x, 2*class_num]
        # vector_maks = mask_ground_truth[..., :]

        ngbbox_yx = ground_truth[..., 0:2] / stride
        ngbbox_yx_round = tf.floor(ngbbox_yx)
        ngbbox_yx_round_int = tf.cast(ngbbox_yx_round, tf.int64)

        # offset_gt = ngbbox_yx - ngbbox_yx_round

        size_gt = ground_truth[..., 2:6] / stride

        keypoints_loss = self._keypoints_loss(keypoints, ngbbox_yx_round_int, ngbbox_y, ngbbox_x, ngbbox_h, ngbbox_w,
                                              class_id, meshgrid_y, meshgrid_x, pshape)

        # get offset from num_g positions
        # offset = tf.gather_nd(offset, ngbbox_yx_round_int)
        size = tf.gather_nd(preg, ngbbox_yx_round_int)
        # offset_loss = tf.reduce_mean(tf.abs(offset_gt - offset))
        size_loss = tf.reduce_mean(tf.square(size_gt - size))
        total_loss = [keypoints_loss, size_loss]
        return total_loss

    def _keypoints_loss(self, keypoints, gbbox_yx, gbbox_y, gbbox_x, gbbox_h, gbbox_w, classid, meshgrid_y, meshgrid_x,
                        pshape):
        sigma = self._gaussian_radius(gbbox_h, gbbox_w, 0.7)
        # [num_g, 1, 1]
        gbbox_y = tf.reshape(gbbox_y, [-1, 1, 1])
        gbbox_x = tf.reshape(gbbox_x, [-1, 1, 1])
        sigma = tf.reshape(sigma, [-1, 1, 1])

        # grid_wall for [num_g, y, x]
        num_g = tf.shape(gbbox_y)[0]
        meshgrid_y = tf.expand_dims(meshgrid_y, 0)
        meshgrid_y = tf.tile(meshgrid_y, [num_g, 1, 1])
        meshgrid_x = tf.expand_dims(meshgrid_x, 0)
        meshgrid_x = tf.tile(meshgrid_x, [num_g, 1, 1])

        # point(3,4) - ground(3,4) key = 1 ; others will reducer as far; [num_g, y, x];
        keyp_penalty_reduce = tf.exp(-((gbbox_y - meshgrid_y) ** 2 + (gbbox_x - meshgrid_x) ** 2) / (2 * sigma ** 2))
        # [h_y, w_x, 1]
        zero_like_keyp = tf.expand_dims(tf.zeros(pshape, dtype=tf.float32), axis=-1)
        reduction = []
        gt_keypoints = []
        for i in range(self.num_class):
            # [num_g, 1]
            exist_i = tf.equal(classid, i+1)  # pass BG CLASS_ID: 0
            # [num_g_of_i, y, x]
            reduce_i = tf.boolean_mask(keyp_penalty_reduce, exist_i, axis=0)
            # [y, x, 1] heat_map for class i , if null class i, product zero_like_map
            reduce_i = tf.cond(
                tf.equal(tf.shape(reduce_i)[0], 0),
                lambda: zero_like_keyp,
                lambda: tf.expand_dims(tf.reduce_max(reduce_i, axis=0), axis=-1)
            )
            reduction.append(reduce_i)
            # according to  class_i index extract gbbox_yx, [num_g_i , 2]
            gbbox_yx_i = tf.boolean_mask(gbbox_yx, exist_i)
            # [y, x, 1]
            gt_keypoints_i = tf.cond(
                tf.equal(tf.shape(gbbox_yx_i)[0], 0),
                lambda: zero_like_keyp,
                lambda: tf.expand_dims(tf.sparse.to_dense(
                    tf.sparse.SparseTensor(gbbox_yx_i, tf.ones_like(gbbox_yx_i[..., 0], tf.float32),
                                           dense_shape=pshape), validate_indices=False), axis=-1)
            )
            gt_keypoints.append(gt_keypoints_i)
        # [y, x, class_num]
        reduction = tf.concat(reduction, axis=-1)
        # [y, x, class_num]
        gt_keypoints = tf.concat(gt_keypoints, axis=-1)
        keypoints_pos_loss = -tf.pow(1. - tf.sigmoid(keypoints), 2.) * tf.math.log_sigmoid(keypoints) * gt_keypoints

        keypoints_neg_loss = -tf.pow(1. - reduction, 4) * tf.pow(tf.sigmoid(keypoints), 2.) * (
                -keypoints + tf.math.log_sigmoid(keypoints)) * (1. - gt_keypoints)
        keypoints_loss = tf.reduce_sum(keypoints_pos_loss) / tf.cast(num_g, tf.float32) + tf.reduce_sum(
            keypoints_neg_loss) / tf.cast(num_g, tf.float32)
        return keypoints_loss

    # def _objvector_loss(self, fcn, objvector, gt_vector_maks,
    #                     classid):
    #     """
    #
    #     :param objvector:
    #     :param gt_vector_maks: [y, x, 3*class_num]
    #     :param gbbox_y:
    #     :param gbbox_x:
    #     :param gbbox_h:
    #     :param gbbox_w:
    #     :param classid:
    #     :param meshgrid_y:
    #     :param meshgrid_x:
    #     :param pshape:
    #     :param stride:
    #     :return:
    #     """
    #
    #     # [h_y, w_x, 2]
    #     zero_t = tf.constant([0.0])
    #     one_t = tf.constant([1.])
    #     # reduction = []
    #     pos_class = []
    #     for i in range(self.num_class):
    #         # [num_g, 1]
    #         exist_i = tf.equal(classid, i+1)
    #         pos_mask = tf.cond(
    #             tf.equal(tf.reduce_sum(tf.dtypes.cast(exist_i, dtype=tf.int32)), 0),
    #             lambda: zero_t,
    #             lambda: one_t
    #         )
    #         pos_class.append(pos_mask)
    #     pos_class = tf.reshape(tf.concat(pos_class, axis=-1), [1, 1, self.num_class])
    #
    #     # positive & negative map
    #     positive_map = tf.dtypes.cast(gt_vector_maks[..., 0::3] == 2, dtype=tf.float32)
    #     pos_num = tf.reduce_sum(positive_map)
    #     negative_map = (tf.dtypes.cast(gt_vector_maks[..., 0::3] == 1, dtype=tf.float32))
    #     neg_num = tf.reduce_sum(negative_map)
    #
    #     positive_y_loss = - positive_map * tf.math.log(tf.keras.activations.sigmoid(fcn))
    #     negative_y_loss = - negative_map * tf.math.log(1. - tf.keras.activations.sigmoid(fcn))
    #
    #     # vector_loss = positive_map * (self._smooth_l1_loss(objvector[..., 0::2], gt_vector_maks[..., 1::3]) +
    #     #                               self._smooth_l1_loss(objvector[..., 1::2], gt_vector_maks[..., 2::3]))
    #     vector_loss = positive_map * (tf.square(objvector[..., 0::2] - gt_vector_maks[..., 1::3]) +
    #                                   tf.square(objvector[..., 1::2] - gt_vector_maks[..., 2::3]))
    #     fcn_loss = tf.constant(0.0)
    #     v_loss = tf.constant(0.0)
    #
    #     if pos_num != 0.0:
    #         fcn_loss += tf.reduce_sum(positive_y_loss) / tf.cast(pos_num, tf.float32)
    #         v_loss += tf.reduce_sum(vector_loss) / tf.cast(pos_num, tf.float32)
    #     if neg_num != 0.0:
    #         fcn_loss += tf.reduce_sum(negative_y_loss) / tf.cast(neg_num, tf.float32)
    #     if pos_num == 0:
    #         print("")
    #         print("pos: ", pos_num, " v_loss: ", v_loss, " fcn_loss: ", fcn_loss)
    #     if neg_num == 0:
    #         print(" ")
    #         print("neg: ", neg_num, " v_loss: ", v_loss, " fcn_loss: ", fcn_loss)
    #     return [fcn_loss, v_loss]

    def _gaussian_radius(self, height, width, min_overlap=0.7):
        a1 = 1.
        b1 = (height + width)
        c1 = width * height * (1. - min_overlap) / (1. + min_overlap)
        sq1 = tf.sqrt(b1 ** 2. - 4. * a1 * c1)
        r1 = (b1 + sq1) / 2.
        a2 = 4.
        b2 = 2. * (height + width)
        c2 = (1. - min_overlap) * width * height
        sq2 = tf.sqrt(b2 ** 2. - 4. * a2 * c2)
        r2 = (b2 + sq2) / 2.
        a3 = 4. * min_overlap
        b3 = -2. * min_overlap * (height + width)
        c3 = (min_overlap - 1.) * width * height
        sq3 = tf.sqrt(b3 ** 2. - 4. * a3 * c3)
        r3 = (b3 + sq3) / 2.

        return tf.reduce_min([r1, r2, r3])

    # def _smooth_l1_loss(self, y_true, y_pred):
    #     """Implements Smooth-L1 loss.
    #     y_true and y_pred are typically: [N, 4], but could be any shape.
    #     """
    #     diff = tf.abs(y_true - y_pred)
    #     less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
    #     loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    #     return loss


class CenterNet:
    def __init__(self, config, name):
        self.name = name
        self.config = config
        assert config.MODEL in ['train', 'infer']
        self.mode = config.MODEL
        self.data_shape = config.IMAGE_SHAPE
        self.data_format = config.DATA_FORMAT
        self.image_size = config.IMAGE_MAX_DIM
        self.num_classes = config.NUM_CLASSES
        self.loss_decay = config.LOSS_DECAY
        self.l2_decay = config.L2_DECAY
        self.batch_size = config.BATCH_SIZE if config.MODEL == 'train' else 1
        self.max_gt_instances = config.MAX_GT_INSTANCES
        self.gt_channel = config.GT_CHANNEL
        self.top_k_results_output = config.DETECTION_MAX_INSTANCES
        self.train_bn = config.TRAIN_BN
        self.score_threshold = config.SCORE_THRESHOLD
        self.is_training = True if config.MODEL == 'train' else False

        if not os.path.exists(name):
            os.mkdir(name)
        self.checkpoint_path = name

        if not os.path.exists(name + "/log"):
            os.mkdir(name + "/log")
        self.log_dir = name + "/log"

        if not os.path.exists(name+"/epoch.txt"):
            file = open(name+"/epoch.txt", 'w')
            file.write("0")
            file.close()

        file = open(name + "/epoch.txt", 'r')
        self.pro_epoch = int(str(file.readline()))
        file.close()
        self._define_inputs()
        self._build_backbone()
        self._build_graph()
        if self.pro_epoch != 0:
            self.load_weight(self.pro_epoch)

    def _define_inputs(self):
        # model inputs: [images, ground_truth, mask_ground_truth]
        shape = self.data_shape
        self.images = tf.keras.Input(shape=shape, dtype=tf.float32)

        if self.mode == 'train':
            gt_shape = [self.max_gt_instances, self.gt_channel]
            self.ground_truth = tf.keras.Input(shape=gt_shape, dtype=tf.float32)
            # mask_shape = [self.image_size/4, self.image_size/4, self.num_classes * 3]
            # self.mask_ground_truth = tf.keras.Input(shape=mask_shape, dtype=tf.float32)

    def _build_backbone(self):
        self.conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=2, padding="same")  # 208, 16
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.bneck1_1 = BottleNeck(in_size=16, exp_size=16, out_size=16, s=1, is_se_existing=False, NL="RE", k=3)  # 208
        self.bneck1_2 = BottleNeck(in_size=16, exp_size=64, out_size=24, s=2, is_se_existing=False, NL="RE", k=3)  # 104

        self.bneck2_1 = BottleNeck(in_size=24, exp_size=72, out_size=24, s=1, is_se_existing=False, NL="RE", k=3)  # 104
        self.bneck2_2 = BottleNeck(in_size=24, exp_size=72, out_size=40, s=2, is_se_existing=True, NL="RE", k=5)  # 52

        self.bneck3_1 = BottleNeck(in_size=40, exp_size=120, out_size=40, s=1, is_se_existing=True, NL="RE", k=5)  # 52
        self.bneck3_2 = BottleNeck(in_size=40, exp_size=120, out_size=40, s=1, is_se_existing=True, NL="RE", k=5)  # 52
        self.bneck3_3 = BottleNeck(in_size=40, exp_size=240, out_size=80, s=2, is_se_existing=False, NL="HS", k=3)  # 26

        self.bneck4_1 = BottleNeck(in_size=80, exp_size=200, out_size=80, s=1, is_se_existing=False, NL="HS", k=3)  # 26
        self.bneck4_2 = BottleNeck(in_size=80, exp_size=184, out_size=80, s=1, is_se_existing=False, NL="HS", k=3)  # 26
        self.bneck4_3 = BottleNeck(in_size=80, exp_size=184, out_size=80, s=1, is_se_existing=False, NL="HS", k=3)  # 26
        self.bneck4_4 = BottleNeck(in_size=80, exp_size=480, out_size=112, s=1, is_se_existing=True, NL="HS", k=3)  # 26
        self.bneck4_5 = BottleNeck(in_size=112, exp_size=672, out_size=112, s=1, is_se_existing=True, NL="HS", k=3)  # 26
        self.bneck4_6 = BottleNeck(in_size=112, exp_size=672, out_size=160, s=2, is_se_existing=True, NL="HS", k=5)  # 13

        self.bneck5_1 = BottleNeck(in_size=160, exp_size=960, out_size=160, s=1, is_se_existing=True, NL="HS", k=5)  # 13
        self.bneck5_2 = BottleNeck(in_size=160, exp_size=960, out_size=160, s=1, is_se_existing=True, NL="HS", k=5)  # 13
        self.conv2 = tf.keras.layers.Conv2D(filters=960, kernel_size=(1, 1), strides=1, padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()


    def _fusion_feature(self):
        inter1 = yolo_block(self.s_32, 512)  # 13*13*1024->(13*13*512,13*13*1024)
        inter1 = yolo_conv2d(inter1, 256, 3, 1)  # # 13*13, 256
        inter1 = upsample_layer(inter1, out_shape=[26, 26])  # 26*26*256
        concat1 = tf.concat([inter1, self.s_16], axis=3)  # 26*26*(256+112)=26*26*368
        inter2 = yolo_block(concat1, 256)  # 26*26*368->26*26*256
        inter2 = yolo_conv2d(inter2, 256, 3, 1)  # 26*26*256->26*26*256
        inter2 = upsample_layer(inter2, out_shape=[52, 52])  # 26*26*256->52*52*256
        concat2 = tf.concat([inter2, self.s_8], axis=3)  # 52*52*(256+40)->52*52*296
        inter3 = yolo_block(concat2, 256)  # 52*52*296->52*52*256
        feature_map = yolo_conv2d(inter3, 256, 3, 1)  # 52*52*256->52*52*128
        return feature_map

    def _detect_head(self, bottom):
        conv1 = self._conv_bn_activation(bottom, 256, 3, 1)
        conv2 = self._conv_bn_activation(conv1, 256, 3, 1)
        conv3 = self._conv_bn_activation(conv2, 256, 3, 1)
        conv4 = self._conv_bn_activation(conv3, 256, 3, 1)
        keypoints = self._conv_activation(conv4, self.num_classes, 3, 1, activation=None)

        conva = self._conv_bn_activation(bottom, 256, 3, 1)
        convb = self._conv_bn_activation(conva, 256, 3, 1)
        convc = self._conv_bn_activation(convb, 256, 3, 1)
        convd = self._conv_bn_activation(convc, 256, 3, 1)
        preg = self._conv_activation(convd, 4, 3, 1, activation=tf.exp)
        return keypoints, preg

    def _build_graph(self):
        x = self.conv1(self.images)
        x = self.bn1(x)
        x = h_swish(x)  # 208
        x = self.bneck1_1(x)  # 208
        x = self.bneck1_2(x)  # 104
        self.s_4 = self.bneck2_1(x)  # 104*104, 24
        x = self.bneck2_2(self.s_4)  # 52
        x = self.bneck3_1(x)  # 52
        self.s_8 = self.bneck3_2(x)  # 52*52, 40
        x = self.bneck3_3(self.s_8)  # 26
        x = self.bneck4_1(x)  # 26
        x = self.bneck4_2(x)  # 26
        x = self.bneck4_3(x)  # 26
        x = self.bneck4_4(x)  # 26
        self.s_16 = self.bneck4_5(x)  # 26*26, 112
        x = self.bneck4_6(self.s_16)  # 13
        x = self.bneck5_1(x)  # 13
        x = self.bneck5_2(x)  # 13
        x = self.conv2(x)  # 13
        x = self.bn2(x)  # 13
        self.s_32 = h_swish(x)  # 13*13, 960

        feature_map = self._fusion_feature()
        stride = 8.0
        keypoints, preg = self._detect_head(feature_map)

        if self.mode == 'train':
            center_loss = CentLoss(self.batch_size, self.num_classes, self.loss_decay)\
                ([keypoints, preg, self.ground_truth])
            inputs = [self.images, self.ground_truth]
            outputs = [keypoints, preg, center_loss]
        # else:
        #     pshape = [tf.shape(offset)[1], tf.shape(offset)[2]]
        #
        #     h = tf.range(0., tf.cast(pshape[0], tf.float32), dtype=tf.float32)
        #     w = tf.range(0., tf.cast(pshape[1], tf.float32), dtype=tf.float32)
        #     # shape of coordinate equals [h_y_num, w_x_mun]
        #     [meshgrid_x, meshgrid_y] = tf.meshgrid(w, h)
        #     meshgrid_y = tf.expand_dims(meshgrid_y, axis=-1)
        #     meshgrid_x = tf.expand_dims(meshgrid_x, axis=-1)
        #     # [y, x, 2]
        #     center = tf.concat([meshgrid_y, meshgrid_x], axis=-1)
        #
        #     # [batch_size, y, x, class_num]
        #     keypoints = tf.sigmoid(keypoints)
        #     print("key points shape: ", np.shape(keypoints))
        #     for i in range(np.shape(keypoints)[0]):
        #         pic_keypoint = keypoints[i]
        #         print("pic_keypoint shape: ", np.shape(pic_keypoint))
        #         pic_offset = offset[i]
        #         pic_size = size[i]
        #         # [y, x, 1] content is class_index_of_max
        #         category = tf.expand_dims(tf.squeeze(tf.argmax(pic_keypoint, axis=-1, output_type=tf.int32)), axis=-1)
        #         # [y, x, 1 + 2(y, x) + 1(index_of_class)=4]
        #         # meshgrid_xyz = tf.concat([tf.zeros_like(category), tf.cast(center, tf.int32), category], axis=-1)
        #         meshgrid_xyz = tf.concat([tf.cast(center, tf.int32), category], axis=-1)
        #         pic_keypoints = tf.gather_nd(pic_keypoint, meshgrid_xyz)
        #         pic_keypoints = tf.squeeze(pic_keypoints)
        #         # [1, y, x, 1(top_value)]
        #         pic_keypoints = tf.expand_dims(pic_keypoints, axis=0)
        #         pic_keypoints = tf.expand_dims(pic_keypoints, axis=-1)
        #         pic_keypoints = tf.reshape(pic_keypoints, (1, tf.shape(offset)[1], tf.shape(offset)[2], 1))
        #
        #         # 3*3 to be peak value
        #         keypoints_peak = self._max_pooling(pic_keypoints, 8, 1)
        #         # mask for each peak_point in each 3*3 area
        #         keypoints_mask = tf.cast(tf.equal(pic_keypoints, keypoints_peak), tf.float32)
        #
        #         pic_keypoints = pic_keypoints * keypoints_mask
        #         # [y*x]
        #         scores = tf.reshape(pic_keypoints, [-1])
        #         # [y*x]
        #         class_id = tf.reshape(category, [-1])
        #         # [(y* x), 2]
        #         bbox_yx = tf.reshape(center + pic_offset, [-1, 2]) * stride
        #         bbox_tlbr = tf.reshape(pic_size, [-1, 4]) * stride
        #         print("shape of tlbr: ", np.shape(bbox_tlbr))
        #
        #         score_mask = scores > self.score_threshold
        #         # results of mask [y*x]
        #         scores = tf.boolean_mask(scores, score_mask)
        #         class_id = tf.boolean_mask(class_id, score_mask)
        #         bbox_yx = tf.boolean_mask(bbox_yx, score_mask)
        #         bbox_hw = tf.boolean_mask(bbox_tlbr, score_mask)
        #         # gt_basic = [gt_y, gt_x, gt_top, gt_left, gt_bot, gt_right, class_ids]
        #         bbox = tf.concat([bbox_yx - bbox_hw[..., 0:2], bbox_yx + bbox_hw[..., 2:4]], axis=-1)
        #         print("shape of bbox: ", np.shape(bbox))
        #         num_select = tf.shape(scores)[0]
        #         select_scores, select_indices = tf.nn.top_k(scores, num_select)
        #         # [1, y*x]
        #         select_scores = tf.expand_dims(select_scores, axis=0)
        #         select_center = tf.expand_dims(tf.gather(bbox_yx, select_indices), axis=0)
        #         select_class_id = tf.expand_dims(tf.gather(class_id, select_indices), axis=0)
        #         select_bbox = tf.expand_dims(tf.gather(bbox, select_indices), axis=0)
        #         print("shape of selected_bbox: ", select_bbox)
        #         # TODO: add list and concat all batch
        #     outputs = [select_scores, select_center, select_bbox, select_class_id, fcn, objvector]
        #     inputs = [self.images]
        self.CenterNetModel = tf.keras.Model(inputs=inputs, outputs=outputs)

    def compile(self):
        """Gets the model ready for training. Adds losses including regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Add L2 Regularization
        reg_losses = self.l2_decay * tf.add_n([tf.nn.l2_loss(var) for var in self.CenterNetModel.trainable_weights])
        self.CenterNetModel.add_loss(lambda: tf.reduce_sum(reg_losses))

        # Optimizer object
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        # optimizer = tf.keras.optimizers.SGD(learning_rate=0.00001)

        self.CenterNetModel.compile(optimizer=optimizer)

    def train_epochs(self, dataset, valset, config, epochs=50):
        self.compile()
        # iter_data = dataset.generator(config.BATCH_SIZE, config.STEPS_PER_EPOCH)
        # val_generator = valset.generator(config.BATCH_SIZE, config.VALIDATION_STEPS)

        epochRec = EpochRecord(self.name)
        callbacks = [
            epochRec,
            tf.keras.callbacks.ProgbarLogger(),
            # tf.keras.callbacks.ReduceLROnPlateau(moniter='val_loss', factor=0.1, patience=2, mode='min', min_lr=1e-7),
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=0, write_graph=True,
                                           write_images=False),
            tf.keras.callbacks.ModelCheckpoint(self.checkpoint_path + "/weights.{epoch:03d}-{loss:.2f}.hdf5", verbose=0, save_weights_only=True)
        ]
        step = int(config.PIC_NUM / config.BATCH_SIZE)
        print("=====ready for model.fit_generator======")
        self.CenterNetModel.fit_generator(
            dataset,
            initial_epoch=self.pro_epoch,
            epochs=epochs,
            max_queue_size=16,
            workers=4,
            steps_per_epoch=step,
            use_multiprocessing=True,
            # validation_data=val_generator,
            # validation_steps=self.config.VALIDATION_STEPS,
            # validation_freq=1,
            callbacks=callbacks
        )

    def test_one_image(self, images, show=False):
        self.is_training = False
        image, window, scale, padding, crop = resize_image(
            images,
            min_dim=self.image_size,
            min_scale=0,
            max_dim=self.image_size,
            mode="square")
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        mean = np.reshape(mean, [1, 1, 3])
        std = np.reshape(std, [1, 1, 3])
        image = (image / 255. - mean) / std
        image = tf.convert_to_tensor(np.expand_dims(image, axis=0))
        pred = self.CenterNetModel.predict(
            image,
            batch_size=1,
            verbose=0,
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False
        )
        # if show:
        #     display_instances(image, centers, boxes, masks, class_ids, class_names,
        #                       scores=None, title="",
        #                       figsize=(16, 16), ax=None,
        #                       show_mask=False, show_bbox=True, show_centers=True,
        #                       colors=None, captions=None)
        return pred

    def load_weight(self, epoch):
        # latest = tf.train.latest_checkpoint(self.checkpoint_path)
        epoch = str(epoch).zfill(3)
        latest = ""
        for filename in os.listdir(self.checkpoint_path):
            root, ext = os.path.splitext(filename)
            if root.startswith('weights.' + epoch) and ext == '.hdf5':
                latest = filename
                break
        self.CenterNetModel.load_weights("./" + self.checkpoint_path + "/" + latest, by_name=True)
        print('load weight', latest, 'successfully')

    def load_pretrained_weight(self, path):
        self.pretrained_saver.restore(self.sess, path)
        print('load pretrained weight', path, 'successfully')

    def _bn(self, bottom):
        bn = tf.keras.layers.BatchNormalization()(bottom)
        return bn

    def _conv_bn_activation(self, bottom, filters, kernel_size, strides, activation=h_swish):
        conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            data_format=self.data_format
        )(bottom)
        bn = self._bn(conv)
        if activation is not None:
            return activation(bn)
        else:
            return bn

    def _conv_activation(self, bottom, filters, kernel_size, strides, activation=h_swish):
        conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            data_format=self.data_format
        )(bottom)
        if activation is not None:
            return activation(conv)
        else:
            return conv


