# Base Info
python version: 3.6.2

# Usage

## trainval_net.py
### vgg16
--weight data/imagenet_weights/vgg16.pth --imdb voc_2007_trainval --imdbval voc_2007_test --iters 70000 --cfg experiments/cfgs/vgg16.yml --net vgg16 --set ANCHOR_SCALES [8,16,32] ANCHOR_RATIOS [0.5,1,2] TRAIN.STEPSIZE [50000]
### resnet50
--weight data/imagenet_weights/res50.pth --imdb voc_2007_trainval --imdbval voc_2007_test --iters 70000 --cfg experiments/cfgs/res50.yml --net res50 --set ANCHOR_SCALES [8,16,32] ANCHOR_RATIOS [0.5,1,2] TRAIN.STEPSIZE [50000]

## demo.py


## train_faster_rcnn.sh
./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc vgg16  
./experiments/scripts/train_faster_rcnn.sh 1 coco res101