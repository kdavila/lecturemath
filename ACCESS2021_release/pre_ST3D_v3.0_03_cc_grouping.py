#============================================================================
# Preprocessing Model for ST3D indexing - V 3.0
#
# Kenny Davila
# - Created:  March 18, 2017
# - Modified: December 31, 2021
#
#============================================================================

import sys
import time
import numpy as np

from AccessMath.data.space_time_struct import SpaceTimeStruct

# from AccessMath.preprocessing.config.parameters import Parameters
# from AccessMath.preprocessing.content.cc_stability_estimator import CCStabilityEstimator
from AccessMath.preprocessing.user_interface.console_ui_process import ConsoleUIProcess
from AM_CommonTools.util.time_helper import TimeHelper


def process_input(process, input_data):
    frame_times, frame_indices, estimator = input_data

    # configuration overrides ...
    if "img_t" in process.params:
        group_min_img_t = float(process.params["img_t"])
    else:
        group_min_img_t = process.configuration.get_float("CC_GROUPING_MIN_IMAGE_THRESHOLD", 0.5)

    cc_group_min_recall = process.configuration.get("CC_GROUPING_MIN_RECALL", 0.0)
    cc_group_min_time_fmeasure = process.configuration.get("CC_GROUPING_MIN_TIME_F_MEASURE", 0.5)
    cc_group_min_time_IOU = process.configuration.get("CC_GROUPING_MIN_TIME_IOU", 0.25)

    cc_stab_max_gap = process.configuration.get_int("CC_STABILITY_MAX_GAP", 85)
    cc_stab_min_times = process.configuration.get_int("CC_STABILITY_MIN_TIMES", 3)

    # remove unstable elements
    # 1) Get original binary frames
    print("Rebuilding binary frames ... ")
    rebuilt_frames = estimator.rebuilt_binary_images()

    # 2) Split CC with large gaps
    # (the threshold is considered in the previous step, but this is applied just in case that
    #  a lower threshold is selected after the initial processing was completed)
    print("Splitting CC with large gap ... ")
    count = estimator.split_stable_cc_by_gaps(cc_stab_max_gap, cc_stab_min_times)
    print("Total CC split: " + str(count))

    # 3) Get the list of stable CC
    print("Computing stable CC")
    stable_idxs = estimator.get_stable_cc_idxs(cc_stab_min_times)

    n_objects = len(estimator.unique_cc_objects)
    n_stable = len(stable_idxs)
    n_raw_objects = estimator.get_raw_cc_count()

    print("Raw CC count: " + str(n_raw_objects))
    print("Unique CC Count: " + str(n_objects))
    print("Stable CC Count: " + str(n_stable))

    # 4) identify overlapping CC
    print("Computing Stable overlapping")

    t_window = process.configuration.get_int("CC_GROUPING_TEMPORAL_WINDOW", 5)
    time_overlapping_cc, total_intersections, all_overlapping_cc = estimator.compute_overlapping_stable_cc(stable_idxs,
                                                                                                           t_window)
    inter_counts = np.array([len(x) for x in time_overlapping_cc])
    hist, bin_edges = np.histogram(inter_counts, 10)

    print("")
    print("Total intersections found: " + str(total_intersections))
    print("Intersection histogram:")
    print(bin_edges)
    print(hist)

    # 5) Use overlapping information to compute groups of CC
    cc_groups, group_idx_per_cc = estimator.compute_groups(stable_idxs, time_overlapping_cc, cc_group_min_recall,
                                                           cc_group_min_time_fmeasure, cc_group_min_time_IOU)
    n_groups = len(cc_groups)
    print("Final count of groups: " + str(n_groups))
    print("Final count of non-empty groups: " + str(sum([1 for x in cc_groups if len(x) > 0])))

    # 6) Get temporal information for groups
    print("Computing ages for groups")
    group_ages, groups_per_frame = estimator.compute_groups_temporal_information(cc_groups)

    # 7) Use temporal information to conflicting groups ...
    # First, check which group cannot exist on the same keyframe (space conflicts)
    print("Computing conflicts between groups")
    conflicts = estimator.compute_conflicting_groups(stable_idxs, all_overlapping_cc, n_groups, group_idx_per_cc)

    # 8) compute the images for each group
    print("Computing images for groups")
    group_images, group_boundaries = estimator.compute_group_images(cc_groups, group_ages, group_min_img_t)

    # 9) generate final binary images ...
    print("Generating output images")
    # save_file_prefix = process.img_dir + "/reconstructed_bin_" + process.current_lecture.title + "_"
    save_file_prefix = None
    clean_binary = estimator.frames_from_groups(cc_groups, group_boundaries, groups_per_frame, group_ages, group_images,
                                                save_file_prefix, cc_stab_min_times, True)

    # Note that here CC Grouping represents a different set of elements than in the previous pipeline
    # also, added redundancy for self-containment and compatibility ...

    # Basically, the reconstructed binary frames after removing background and unstable CC
    cc_reconstructed = (frame_times, frame_indices, clean_binary)

    # Only temporal information and conflicts, for temporal segmentation
    cc_conflict_info = (group_ages, conflicts)

    # The final 3D structure (original frame info + CC Groups (Ages, Shape, Location)
    # for indexing and summary generation
    st3D = SpaceTimeStruct(frame_times, frame_indices, estimator.height, estimator.width,
                           group_ages, group_images, group_boundaries)

    return [cc_reconstructed, cc_conflict_info, st3D]


def main():
    # usage check
    if not ConsoleUIProcess.usage_with_config_check(sys.argv):
       return

    process = ConsoleUIProcess.FromConfigPath(sys.argv[1], sys.argv[2:], "CC_STABILITY_OUTPUT",
                                              ["CC_RECONSTRUCTED_OUTPUT", "CC_CONFLICTS_OUTPUT", "CC_ST3D_OUTPUT"])
    if not process.initialize():
       return

    start_time = time.time()
    process.start_input_processing(process_input)
    end_time = time.time()

    print("Total time: " + TimeHelper.secondsToStr(end_time - start_time))
    print("Finished")


if __name__ == "__main__":
    main()
