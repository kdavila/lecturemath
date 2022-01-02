## Welcome to LectureMath (IEEE ACCESS 2021 Release)

The files in here are for the paper *FCN-LectureNet: Extractive Summarization of Whiteboard and Chalkboard Lecture Videos* (IEEE ACCESS 2021)

![alt text](https://raw.githubusercontent.com/adaniefei/Other/images/Overall-Arch.png?raw=true "overall-arch")

![alt text](https://raw.githubusercontent.com/adaniefei/Other/images/FCN-LectureNet-Arch.png "fcn-lecturenet-arch")

Network Training Pipeline
OK - lecturenet_train_00_pretrain_reconstruction.py
OK - lecturenet_train_01_pretrain_text_detector.py
OK - lecturenet_train_02_train_binarizer.py

Summarization Pipeline
OK - python pre_ST3D_v3.0_01_binarize.py
OK - python pre_ST3D_v3.0_02_cc_analysis.py
OK - python pre_ST3D_v3.0_03_cc_grouping.py
OK - python pre_ST3D_v3.0_04_vid_segmentation.py
OK - python pre_ST3D_v3.0_05_generate_summary.py

Evaluation Pipeline
OK - lecturenet_eval_pretrain_text_detector.py
OK - lecturenet_eval_segments.py
OK - lecturenet_eval_keyframe_bin.py
OK - test_FCN_Binarizer

Tools
OK - lecturenet_data_00_prepare_binary_text_masks.py
OK - TEXT_ICDAR2017_COCOText_prepare.py
OK - TEXT_dataset_validate_files.py
OK - vis_gt_invervals.py


#### News (Updated: 7/20/2021):
The code and data for our paper will be available here soon. 
