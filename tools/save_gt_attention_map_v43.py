import os
import cv2
import numpy as np
import _init_paths
from model.test import im_detect
from nets.resnet_v1 import resnetv1
from model.config import cfg
import global_var
import torch
from torch.autograd import Variable
from model.nms_wrapper import nms
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import pickle
CLASSES = ('__background__',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor',
                   'aeroplane_truncated', 'bicycle_truncated', 'bird_truncated', 'boat_truncated',
                   'bottle_truncated', 'bus_truncated', 'car_truncated', 'cat_truncated', 'chair_truncated',
                   'cow_truncated', 'diningtable_truncated', 'dog_truncated', 'horse_truncated',
                   'motorbike_truncated', 'person_truncated', 'pottedplant_truncated',
                   'sheep_truncated', 'sofa_truncated', 'train_truncated', 'tvmonitor_truncated'
                   )
all_attention_maps = {}

"""load ground_truth"""
def parse_rec(filename):
  """ Parse a PASCAL VOC xml file """
  tree = ET.parse(filename)
  objects = []
  for obj in tree.findall('object'):
    obj_struct = {}
    obj_struct['name'] = obj.find('name').text
    obj_struct['pose'] = obj.find('pose').text
    obj_struct['truncated'] = int(obj.find('truncated').text)
    obj_struct['difficult'] = int(obj.find('difficult').text)
    bbox = obj.find('bndbox')
    obj_struct['bbox'] = [int(bbox.find('xmin').text),
                          int(bbox.find('ymin').text),
                          int(bbox.find('xmax').text),
                          int(bbox.find('ymax').text)]
    objects.append(obj_struct)
  return objects

# read list of images
imagesetfile = '/data/zhbli/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
annopath = '/data/zhbli/VOCdevkit/VOC2007/Annotations/{:s}.xml'
cachefile = '/home/zhbli/temp/voc07_trainval_anno_cache.pkl'
with open(imagesetfile, 'r') as f:
    lines = f.readlines()
    imagenames = [x.strip() for x in lines]

if not os.path.isfile(cachefile):
    recs = {}
    for i, imagename in enumerate(imagenames):
      recs[imagename] = parse_rec(annopath.format(imagename))
      if i % 100 == 0:
        print('Reading annotation for {:d}/{:d}'.format(
          i + 1, len(imagenames)))
    # save
    print('Saving cached annotations to {:s}'.format(cachefile))
    with open(cachefile, 'wb') as f:
        pickle.dump(recs, f)
else:
    # load
    with open(cachefile, 'rb') as f:
        try:
            recs = pickle.load(f)
        except:
            recs = pickle.load(f, encoding='bytes')

"""create net"""
net = resnetv1(num_layers=50)
net.create_architecture(41,
                        tag='default', anchor_scales=[8, 16, 32])
saved_model = '/home/zhbli/Project/fast-rcnn/output/res50/voc_2007_trainval/default/res50_faster_rcnn_iter_70000.pth'
net.load_state_dict(torch.load(saved_model))

net.eval()
net.cuda()

# v4.0
# hook the feature extractor
finalconv_name = 'resnet'
features_blobs = [-1]  # shape shoule be [2048, 7, 7]

def hook_feature(module, input, output):
    features_blobs[0] = output.data.cpu().numpy()

net._modules.get(finalconv_name)._modules.get('layer4').register_forward_hook(hook_feature)
# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-4].data.cpu().numpy()) # shape = [41, 2048]

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    bz, nc, h, w = feature_conv.shape
    assert len(class_idx) == 1, 'err: assert len(class_idx) == 1'
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
    return cam

"""handle every ground_truth"""
for i in range(len(recs)):
# for every img
    img = cv2.imread('/data/zhbli/VOCdevkit/VOC2007/JPEGImages/{:s}.jpg'.format(imagenames[i]))
    assert img is not None, "fail to load img"
    print('handling img {:s}'.format(imagenames[i]))
    all_attention_maps[imagenames[i]] = []
    for j in range(len(recs[imagenames[i]])):
    # for every gt
        print('testing img_{:s}, gt_{:d}'.format(imagenames[i], j))
        difficult = recs[imagenames[i]][j]['difficult']
        if difficult:
            continue
        class_name = recs[imagenames[i]][j]['name']
        bbox = recs[imagenames[i]][j]['bbox']

        """resize roi"""
        im_shape = img.shape
        im_size_min = min(im_shape[0:2])
        im_size_max = max(im_shape[0:2])
        for target_size in cfg.TEST.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
                im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        global_roi = np.asarray([0, bbox[0], bbox[1], bbox[2], bbox[3]])
        global_roi = global_roi * im_scale
        global_roi = Variable(torch.from_numpy(global_roi).type(torch.cuda.FloatTensor).unsqueeze(0))
        global_var.global_roi = global_roi
        """resize roi"""

        scores, _ = im_detect(net, img)
        idx = np.argmax(scores, 1).squeeze()
        if CLASSES[idx] != class_name:
            print('err: {:s}'.format(imagenames[i]))
        true_idx = CLASSES.index(class_name)

        # v4.0
        CAM = returnCAM(features_blobs[0], weight_softmax, [true_idx])
        # heatmap = cv2.resize(CAM, (bbox[2] - bbox[0], bbox[3] - bbox[1]))
        all_attention_maps[imagenames[i]].append(CAM)

save_file_name = '/data/zhbli/VOCdevkit/results/VOC2007/CAM/all_attention_maps.pkl'
save_file = open(save_file_name, 'wb')
pickle.dump(all_attention_maps, save_file)
exit()
