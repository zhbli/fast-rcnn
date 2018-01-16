# v0.0
Original version: pytorch-faster-rcnn

# v0.1
Change dictionary config to run the code successfully.

# v1.0  
Train net with ground_truth boxes:  
- Regard only classify_ loss as total loss.
- Ignore RPN.

# v2.0
Detect any roi.

Change list:
- [Add] test_any_roi.py
- [Add] global_var.py
- [Modify] network.py

# v3.0
Train truncated objects.

Change list:
- [Add_func] genarate_truncated_rois
- [Modify_func] _sample_rois_manually
- [Modify_file] imdb.py

# v4.0
Generate attention heat map.

Usage: Run `test_any_roi.py`

Change list:
- [Modify_file] test_any_roi.py