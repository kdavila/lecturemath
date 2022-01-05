## Welcome to LectureMath (IEEE ACCESS 2021 Release)

The files in here are for the paper [*FCN-LectureNet: Extractive Summarization of Whiteboard and Chalkboard Lecture Videos*](https://ieeexplore.ieee.org/abstract/document/9494351) (IEEE ACCESS 2021). These files are mainly grouped into four sections:

[FCN-LectureNet Training](#fcn-lecturenet-training)

[Summarization Pipeline](#summarization-pipeline)

[Evaluation Pipeline](#evaluation-pipeline)

[Other Tools](#other-tools)

The config file [config/FCN_LectureNet.conf](https://github.com/kdavila/lecturemath/blob/master/ACCESS2021_release/configs/FCN_LectureNet.conf) stores all parameters needed for these scripts. 

The **lecuture_data** folder includes default directories for (pre)training input data and saving outputs of the framework. 
In addition, **db_LectureMath.xml** contains the meta data of all 34 videos in LectureMath dataset, and **video_edited_gt_34.json** includes the binary annotation about unedited / edited videos in LectureMath dataset.



## FCN-LectureNet Training

![alt text](https://raw.githubusercontent.com/adaniefei/Other/images/FCN-LectureNet3branches.png?raw=true "fcn-lecturenet-arch")

This section is for the training of *FCN-LectureNet*. As described in the paper, there are three separated branches: background estimation, text-mask estimation, and binarization of the resulting image from the previous two branches. The training is done by the following script:

    python lecturenet_train_02_train_binarizer.py [path of config file]

To improve the performance, background estimation is pretrained via reconstructing median filtered image (**Med-PT**). 

    python lecturenet_train_00_pretrain_reconstruction.py [path of config file]

The text-mask estimation branch is pretrained from pixel-level text detection (**TD-PT**). 

    python lecturenet_train_01_pretrain_text_detector.py [path of config file]

To pretrain text-mask branch from the pretrained reconstruction model, *FCN_BINARIZER_PRETRAIN_USE_RECONSTRUCTION_OUTPUT* needs to be set True. (**Med-PT + TD-PT**). 

Similarly, to adopt the pretrained models while training *FCN-LectureNet*, the following parameters need to set correctly in relative scripts based on the pretraining mode. 

    FCN_BINARIZER_TRAIN_USE_PRETRAIN_OUTPUT
    FCN_BINARIZER_TRAIN_FROM_RECONSTRUCTION_PRETRAIN
    
All models are saved in **output/models/** as default.

## Summarization Pipeline

Scripts in this section are for the entire pipeline of lecture video summarization.

<p align="center">
  <img src="https://raw.githubusercontent.com/adaniefei/Other/images/Overall-Arch.png">
  Figure 1: Overall architecture of our lecture video summarization approach.
</p>

With pretrained models from [previous section](#fcn-lecturenet-training), the framework starts from video frames sampling and binarizing the extracted handwritten content. 

     python pre_ST3D_v3.0_01_binarize.py [path of config file] [other parameters]

The config file provides parameters `FCN_BINARIZER_NET_*` that are used to create and load models. The default video frame sampling is `1 FPS`, and the binary outputs are saved in **output/temporal/** with the prefix `tempo_binary_`. 

>>

These outputs are then analyzed to identify unique and stable Connected Components (CCs) to represent hand written content, and these CCs are grouped into tracklets that define the timeline of each group of handwritting symbols. 
     
     python pre_ST3D_v3.0_02_cc_analysis.py [path of config file] [other parameters]
     python pre_ST3D_v3.0_03_cc_grouping.py [path of config file] [other parameters]
     
In `pre_ST3D_v3.0_03_cc_grouping.py`, three types of CCs are generated:
     
     CC_RECONSTRUCTED_OUTPUT = tempo_bin_reconstructed_
     CC_CONFLICTS_OUTPUT = tempo_cc_conflicts_
     CC_ST3D_OUTPUT = tempo_cc_ST3D_
     
These CCs are used for semantic lecture video segmentation. In `python pre_ST3D_v3.0_04_vid_segmentation.py`, there are three vide segmentation options:

- SUMS: It uses sliding window to find video keyframes that have the maximum content. [This method](https://ieeexplore.ieee.org/document/4351897) is from one of the baselines in the [previous work](https://www.cs.rit.edu/~rlaz/files/Kenny_ICDAR_2017.pdf).   
- CC Conflicts: It splits video via minimizing the CCs conflict. This method is a variation of the [previous work](https://www.cs.rit.edu/~rlaz/files/Kenny_ICDAR_2017.pdf).
- Deletion Events: As described in the paper, this is the method that segments lecture videos via detecting deletion events of major content removal.
     
After segmenting lecture videos, the final video summary is represented by generating keyframe for each video segment using `pre_ST3D_v3.0_05_generate_summary.py
`.

## Evaluation Pipeline

## Other Tools

---

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
