## Welcome to LectureMath (ICPRW DL-HAU 2020 Release)
The files in "[DLHAU2020_release](https://github.com/kdavila/lecturemath/tree/master/DLHAU2020_release)" are for the paper Skeleton-based Methods for Speaker Action Classification on Lecture Videos (ICPRW DL-HAU 2020)

#### whiteboards_2020:
- pose
   - original_accessmath: this folder contains the pose data extracted by four pose estimators for the original videos of AccessMath training set.
   - youtube_lecturemath: this folder contains the pose data extracted by four pose estimators for the youtube videos of AccessMath training set.
   - lecture_math: this folder contains the link to access the pose data extracted by [AlphaPose](https://github.com/adaniefei/AlphaPose) for the youtube videos of LectureMath dataset.
- output
   - action_segments_csv.zip: this file contains data of the action segment information for both training/testing videos of LectureMath dataset.
   - annotations.zip: this file contains action labels for all videos of LectureMath dataset.
- conf_lectureMath.conf: this file is for running the code of [AccessMath Pose](https://github.com/adaniefei/AccessMath_Pose) that is used as the baseline method in this paper.
- db_LectureMath.xml: this file contains video information (e.g. video ID, original video name, youtube link) in LectureMath dataset.


#### 2S-AGCN.zip: 
This file contains config files and some code for 2s-AGCN in experiment II & III. The original 2S-AGCN code can be accessed from [link](https://github.com/lshiwjx/2s-AGCN).

#### video_metrics.json: 
As videos can be re-enconded and re-sampled, it might be possible that their overall number of frames changes. For example, we observed a difference between the frame counts of the original AccessMath videos and their online versions. It might also be convenient to re-encode videos recorded at different frame rate to a standarized frame rate in order to keep the duration of a fixed number of frame a constant across all videos.  Since the final frame rates used at run time might be different from those observed at annotation time, we introduce the usage of a JSON file which summarizes the frame counts for each video in the collection. During run time, this file is used by multiple scripts that will dynamically adjust the ground truth from their original frame count to the frame count observed in the available version of the video (while also assuming that their overall length still remains the same). We have included a file named "video_metrics.json" which represents the count of frames for the versions of each video used in our experiments.

#### LectureMath_video_info.xlsx:
This file contains the basic information including IDs and YouTube links for videos in LectureMath dataset. 
