## Welcome to LectureMath (IEEE ACCESS 2021 Release)

The files in here are for the paper [*FCN-LectureNet: Extractive Summarization of Whiteboard and Chalkboard Lecture Videos*](https://ieeexplore.ieee.org/abstract/document/9494351) (IEEE ACCESS 2021). These files are mainly grouped into four sections:

[FCN-LectureNet Training](#fcn-lecturenet-training)

[Summarization Pipeline](#summarization-pipeline)

[Evaluation Pipeline](#evaluation-pipeline)

[Other Tools](#other-tools)

The config file [config/FCN_LectureNet.conf](https://github.com/kdavila/lecturemath/blob/master/ACCESS2021_release/configs/FCN_LectureNet.conf) stores all parameters needed for these scripts. 

The **lecuture_data** folder includes default directories for (pre)training input data and saving outputs of the framework. 
In addition, **db_LectureMath.xml** contains the meta data of all 34 videos in LectureMath dataset, and **video_edited_gt_34.json** includes the binary annotation about unedited / edited videos in LectureMath dataset. 

In addition, the *LectureMath annotations* used in IEEE Access paper and the *trained model* of FCN-LectureNet can be downloaded in [Data Release](#data-release).


### News:
- The annotations and trained model are now accessible in [Data Release](#data-release). (Updated: 03/14/2022)
- The code and data for our paper have been made available here. (Updated: 01/07/2022)


## FCN-LectureNet Training

<p align="center">
<img src="https://raw.githubusercontent.com/adaniefei/Other/images/FCN-LectureNet3branches.png" width="700" height="685">
</p>
<p align="center">Figure 1: Architecture of the three branches of FCN-LectureNet</p>

This section is for the training of *FCN-LectureNet*. As described in the paper, there are three separated branches: background estimation, text-mask estimation, and binarization of the resulting image from the previous two branches. The training is done by the following script:

    python lecturenet_train_02_train_binarizer.py [path of the config file]

To improve the performance, background estimation is pretrained via reconstructing median filtered image (**Med-PT**). 

    python lecturenet_train_00_pretrain_reconstruction.py [path of the config file]

The text-mask estimation branch is pretrained from pixel-level text detection (**TD-PT**). 

    python lecturenet_train_01_pretrain_text_detector.py [path of the config file]

To pretrain text-mask branch from the pretrained reconstruction model, *FCN_BINARIZER_PRETRAIN_USE_RECONSTRUCTION_OUTPUT* needs to be set True. (**Med-PT + TD-PT**). 

Similarly, to adopt the pretrained models while training *FCN-LectureNet*, the following parameters need to set correctly in relative scripts based on the pretraining mode. 

    FCN_BINARIZER_TRAIN_USE_PRETRAIN_OUTPUT
    FCN_BINARIZER_TRAIN_FROM_RECONSTRUCTION_PRETRAIN
    
All models are saved in **output/models/** as default.

## Summarization Pipeline

Scripts in this section are for the entire pipeline of lecture video summarization.

<p align="center">
  <img src="https://raw.githubusercontent.com/adaniefei/Other/images/Overall-Arch.png">
</p>
<p align="center"> Figure 2: Overall pipeline of our lecture video summarization approach.</p>

With pretrained models from [previous section](#fcn-lecturenet-training), the framework starts from video frames sampling and binarizing the extracted handwritten content. 

     python pre_ST3D_v3.0_01_binarize.py [path of the config file] [other parameters]

The config file provides parameters `FCN_BINARIZER_NET_*` that are used to create and load models. The default video frame sampling is `1 FPS`, and the binary outputs are saved in **output/temporal/** with the prefix `tempo_binary_`. 

>>

These outputs are then analyzed to identify unique and stable Connected Components (CCs) to represent handwritten content, and these CCs are grouped into tracklets that define the timeline of each group of handwritting symbols. 
     
     python pre_ST3D_v3.0_02_cc_analysis.py [path of the config file] [other parameters]
     python pre_ST3D_v3.0_03_cc_grouping.py [path of the config file] [other parameters]
     
In `pre_ST3D_v3.0_03_cc_grouping.py`, three types of CCs are generated:
     
     CC_RECONSTRUCTED_OUTPUT = tempo_bin_reconstructed_
     CC_CONFLICTS_OUTPUT = tempo_cc_conflicts_
     CC_ST3D_OUTPUT = tempo_cc_ST3D_
     
These CCs are used for semantic lecture video segmentation. In `python pre_ST3D_v3.0_04_vid_segmentation.py`, there are three video segmentation options:

- SUMS: It uses sliding window to find video keyframes that have the maximum content. [This method](https://ieeexplore.ieee.org/document/4351897) is from one of the baselines in the [previous work](https://www.cs.rit.edu/~rlaz/files/Kenny_ICDAR_2017.pdf).   
- CC Conflicts: It splits video via minimizing the CCs conflict. This method is a variation of the [previous work](https://www.cs.rit.edu/~rlaz/files/Kenny_ICDAR_2017.pdf).
- Deletion Events: As described in the paper, this is the method that segments lecture videos via detecting deletion events of major content removal.
     
After segmenting lecture videos, the final video summary is represented by generating keyframe for each video segment using `pre_ST3D_v3.0_05_generate_summary.py
`.

## Evaluation Pipeline

To evaluate the generated video segmentations, unedited and edited videos can be considered separately as differnt segmentation methods have different performances on these two types. By default, video segmentation data is saved at `lecture_data/output/temporal/tempo_intervals_**`.

    python lecturenet_eval_segments.py [path to the config file] [dataset split] [(optional)video_edited_gt]
    
The other script `lecturenet_eval_keyframe_bin.py` is used to evaluate the performance of FCN-LectureNet with different pretraining settings. This is evaluated using the groud truth keyframes of LectureMath dataset.

     python lecturenet_eval_keyframe_bin.py [path to the config file] [path to the model] [dataset split]

Both evaluation scripts use the following parameter valus to select dataset type.

    # For [dataset split]: 
    # 1 - training set
    # 0 - testing set

## Other Tools
- `test_FCN_Binarizer.py` generates binary output of a given input image using any FCN-Lecture model and the corresponding config file.
- `lecturenet_data_00_prepare_binary_text_masks.py` generates binary mask of text region for general dataset (e.g. LSVT or ART datasets) that contains a json file of text region annotations.
- `TEXT_ICDAR2017_COCOText_prepare.py` generates binary mask of text region and split the COCO-Text dataset into **training**, **validation**, and **testing** sets.
- `TEXT_dataset_validate_files.py` checks input images, and it will report those ones that:
    - can't be loaded correctly (don't have EXIF orientation info)
    - either the height or the width of the image is less than 256.
- `vis_gt_intervals.py` can be used to generate visulization of video segmentations. The output is similar to the plot below. Every pair of two consecutive vertical red lines represent a video segmentation. 
    <p align="center">
    <img src="https://raw.githubusercontent.com/adaniefei/Other/images/video_interval.jpg" alt="vis of video segmentations" width="640" height="480">
    </p>


## Data Release

The LectureMath annotations used in IEEE Access paper and the trained model of FCN-LectureNet can be downloaded from following links. 

| Name  | Download |
| :---         |     :---:      |
| LectureMath v1.1 | [Link](https://www.dropbox.com/s/5ejyfmeqbr2r2jk/IEEE_access_data_release.zip?dl=0)  |
| FCN-LectureNet Model  | [V34](https://www.dropbox.com/s/ea0266hm1vkcjwc/LectureNet_model_BIN_V34_final.dat?dl=0)  |


### How to use annotations
In *IEEEAccess Annotations*, each video has one folder and one xml file including anotation details, both named as *LectureMath_[video id]*.

    LectureMath_[video id] folder:
        - keyframes: PNG images of video keyframe summaries. Each image is named using its frame number.
        - binary: the binarized version of images from *keyframes* folder. These binary images only preserve hand written content.
        - portions: images of content-wise regions on each binarized keyframe.
        - segments.xml: including keyframe indices and start/end frame number of video segments.
        - portions.xml: indluding portion bounding boxes on each keyframe.
        - unique_ccs.xml: describing the groups of unique connected components in the video. That is, grouping individual connected 
          components across keyframes if they represent the same symbol.
    
    LectureMath_[video id].xml
        - including keyframe indices, start/end frame number of video segments, and speaker actions. Each speaker action is annotated 
          with action label and the start/end frame number of the action. 

**Note**: the \<Polygon\>-\<Points\> in the speaker \<VideoObject\> have not been properly set to the true locations of the speaker and should be ignored.
        
## Citing FCN-LectureNet
If you use FCN-LectureNet in your research or wish to refer to the results published in the paper, please use the following BibTeX entry.

    @ARTICLE{9494351,
    author={Davila, Kenny and Xu, Fei and Setlur, Srirangaraj and Govindaraju, Venu},
    journal={IEEE Access}, 
    title={FCN-LectureNet: Extractive Summarization of Whiteboard and Chalkboard Lecture Videos}, 
    year={2021},
    volume={9},
    number={},
    pages={104469-104484},
    doi={10.1109/ACCESS.2021.3099427}}
    
        
    



