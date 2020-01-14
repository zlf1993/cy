from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils import tfrecord_voc_utils as voc_utils
import numpy as np
import CenterNet as net
import os
from coco.coco import CenterNetCocoConfig, CocoDataset, SequenceData
from PIL import Image

# Set Coco
ROOT_DIR = os.path.abspath("../")
COCO_DIR = ROOT_DIR + "/coco2014"
Config = CenterNetCocoConfig()
Config.display()

dataset = CocoDataset()
dataset.load_coco(Config, COCO_DIR, "train", class_ids=[1, 17, 18])
dataset.prepare()
print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))

# valset = CocoDataset()
# valset.load_coco(Config, COCO_DIR, "val", class_ids=[1, 17, 18])
# valset.prepare()
# print("Image Count: {}".format(len(valset.image_ids)))
# print("Class Count: {}".format(valset.num_classes))

centernet = net.CenterNet(Config, "Creature")
seqdata = SequenceData(Config.PIC_NUM, Config.BATCH_SIZE, dataset, 0)
centernet.train_epochs(seqdata, None, Config, 50)
# image = Image.open('COCO_val2014_000000000761.jpg')
# image_array = np.array(image)
# predict = centernet.test_one_image(image_array)
# print(predict)

# centernet.load_weight('./centernet/test-8350')
# centernet.load_pretrained_weight('./centernet/test-8350')

# for i in range(epochs):
#     print('-'*25, 'epoch', i, '-'*25)
#     if i in reduce_lr_epoch:
#         lr = lr/10.
#         print('reduce lr, lr=', lr, 'now')
#     mean_loss = centernet.train_one_epoch(lr)
#     print('>> mean loss', mean_loss)
#     centernet.save_weight('latest', './centernet/test')            # 'latest', 'best

# img = io.imread('000026.jpg')
# img = transform.resize(img, [384,384])
# img = np.expand_dims(img, 0)
# result = centernet.test_one_image(img)
# id_to_clasname = {k:v for (v,k) in classname_to_ids.items()}
# scores = result[0]
# bbox = result[1]
# class_id = result[2]
# print(scores, bbox, class_id)
# plt.figure(1)
# plt.imshow(np.squeeze(img))
# axis = plt.gca()
# for i in range(len(scores)):
#     rect = patches.Rectangle((bbox[i][1],bbox[i][0]), bbox[i][3]-bbox[i][1],bbox[i][2]-bbox[i][0],linewidth=2,edgecolor='b',facecolor='none')
#     axis.add_patch(rect)
#     plt.text(bbox[i][1],bbox[i][0], id_to_clasname[class_id[i]]+str(' ')+str(scores[i]), color='red', fontsize=12)
# plt.show()
