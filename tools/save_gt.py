import xml.etree.ElementTree as ET
import pickle
import os
import cv2

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
"""end: load ground_truth"""

"""save gt_info into txt"""

"""end: save gt_info into txt"""
file = open('/data/zhbli/VOCdevkit/results/VOC2007/gt_size.txt','w')
for i in range(len(recs)):
# for every img
    for j in range(len(recs[imagenames[i]])):
        class_name = recs[imagenames[i]][j]['name']
        bbox = recs[imagenames[i]][j]['bbox']
        truncated = recs[imagenames[i]][j]['truncated']
        difficult = recs[imagenames[i]][j]['difficult']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        size = width * height
        result = '{:s}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:s}\n'.format(class_name, size, width, height, truncated, difficult, imagenames[i])
        # [class size width height truncated difficult img_name] eg: 'persion 256 16 16 1 0 000034'
        file.write(result)
file.close()
# print('didnot run program fully')
# exit()
"""handle every ground_truth"""
for i in range(len(recs)):
# for every img
    img = cv2.imread('/data/zhbli/VOCdevkit/VOC2007/JPEGImages/{:s}.jpg'.format(imagenames[i]))
    assert img is not None, "fail to load img"
    for j in range(len(recs[imagenames[i]])):
    # for every gt
        print('testing img_{:s}, gt_{:d}'.format(imagenames[i], j))
        class_name = recs[imagenames[i]][j]['name']
        bbox = recs[imagenames[i]][j]['bbox']
        truncated = recs[imagenames[i]][j]['truncated']
        difficult = recs[imagenames[i]][j]['difficult']
        size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        result = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        if not os.path.exists('/data/zhbli/VOCdevkit/results/VOC2007/gt/{:s}'.format(class_name)):
            os.mkdir('/data/zhbli/VOCdevkit/results/VOC2007/gt/{:s}'.format(class_name))
        if not os.path.exists('/data/zhbli/VOCdevkit/results/VOC2007/gt/{:s}/truncated'.format(class_name)):
            os.mkdir('/data/zhbli/VOCdevkit/results/VOC2007/gt/{:s}/truncated'.format(class_name))
        if not os.path.exists('/data/zhbli/VOCdevkit/results/VOC2007/gt/{:s}/untruncated'.format(class_name)):
            os.mkdir('/data/zhbli/VOCdevkit/results/VOC2007/gt/{:s}/untruncated'.format(class_name))
        if not os.path.exists('/data/zhbli/VOCdevkit/results/VOC2007/gt/{:s}/untruncated/easy'.format(class_name)):
            os.mkdir('/data/zhbli/VOCdevkit/results/VOC2007/gt/{:s}/untruncated/easy'.format(class_name))
        if not os.path.exists('/data/zhbli/VOCdevkit/results/VOC2007/gt/{:s}/untruncated/difficult'.format(class_name)):
            os.mkdir('/data/zhbli/VOCdevkit/results/VOC2007/gt/{:s}/untruncated/difficult'.format(class_name))
        if not os.path.exists('/data/zhbli/VOCdevkit/results/VOC2007/gt/{:s}/truncated/easy'.format(class_name)):
            os.mkdir('/data/zhbli/VOCdevkit/results/VOC2007/gt/{:s}/truncated/easy'.format(class_name))
        if not os.path.exists('/data/zhbli/VOCdevkit/results/VOC2007/gt/{:s}/truncated/difficult'.format(class_name)):
            os.mkdir('/data/zhbli/VOCdevkit/results/VOC2007/gt/{:s}/truncated/difficult'.format(class_name))

        if truncated == True:
            if difficult == True:
                cv2.imwrite(
                    '/data/zhbli/VOCdevkit/results/VOC2007/gt/{:s}/truncated/difficult/{:s}_{:d}.jpg'.format(class_name, imagenames[i], j),
                    result)
            else:
                cv2.imwrite(
                    '/data/zhbli/VOCdevkit/results/VOC2007/gt/{:s}/truncated/easy/{:s}_{:d}.jpg'.format(class_name,
                                                                                                   imagenames[i], j),
                    result)
        else:
            if difficult == True:
                cv2.imwrite(
                    '/data/zhbli/VOCdevkit/results/VOC2007/gt/{:s}/untruncated/difficult/{:s}_{:d}.jpg'.format(class_name, imagenames[i], j),
                    result)
            else:
                cv2.imwrite(
                    '/data/zhbli/VOCdevkit/results/VOC2007/gt/{:s}/untruncated/easy/{:s}_{:d}.jpg'.format(class_name,
                                                                                                   imagenames[i], j),
                    result)

