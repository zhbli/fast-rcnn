# Usage

## trainval_net.py
--weight data/imagenet_weights/vgg16.pth --imdb voc_2007_trainval --imdbval voc_2007_test --iters 70000 --cfg experiments/cfgs/vgg16.yml --net vgg16 --set ANCHOR_SCALES [8,16,32] ANCHOR_RATIOS [0.5,1,2] TRAIN.STEPSIZE [50000]