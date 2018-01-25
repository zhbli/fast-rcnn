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

# v4.1
Save every ground_truth's attention_heat_map.

Usage: Run `save_gt_attention_map.py`

Change list:
- [Add_file] save_gt_attention_map.py

# v4.2
Save every ground_truth.

Usage: Run `save_gt.py`

Change list:
- [Add_file] save_gt.py

# v4.3
Save every ground_truth's attention_heat_map.
Result will be saved in .pkl file.

Diff with v4.1:
- Just save the heat map value, will not save the original image.
- Donot normalize the heat map value to [0, 255].

Usage: Run `save_gt_attention_map_v43.py`

Change list:
- [Add_flie] save_gt_attention_map_v43.py