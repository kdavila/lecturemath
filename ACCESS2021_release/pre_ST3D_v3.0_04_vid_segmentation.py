
import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt

from AccessMath.preprocessing.user_interface.console_ui_process import ConsoleUIProcess
from AccessMath.data.space_time_struct import SpaceTimeStruct

from AccessMath.preprocessing.content.helper import Helper
from AccessMath.preprocessing.content.video_segmenter import VideoSegmenter

from AM_CommonTools.util.time_helper import TimeHelper

def process_input(process, input_data):
    segmentation_method = process.configuration.get_int("VIDEO_SEGMENTATION_METHOD", 3)

    if segmentation_method == 3 or segmentation_method == 2:
        # Conflict Minimization or Delete Events
        frame_times, frame_indices, compressed_frames = input_data[0]
    else:
        # SUMS method
        frame_times, frame_indices, compressed_frames = input_data

    debug_mode = True

    # 1) decompress all images ...
    print("Decompressing input...")
    if debug_mode:
        all_binary = Helper.decompress_binary_images(compressed_frames)
    else:
        all_binary = Helper.decompress_binary_images(compressed_frames[:1])
        # all_binary = None

    # 2) Computing all sums...
    if segmentation_method == 1 or debug_mode:
        print("Computing sums...")
        all_sums = VideoSegmenter.compute_binary_sums(all_binary)
    else:
        all_sums = None

    # 3) Getting the intervals ...
    if segmentation_method == 3:
        # inputs ....
        group_ages, conflicts = input_data[1]
        st3D = input_data[2]

        assert isinstance(st3D, SpaceTimeStruct)

        # get config parameters for DEL EVENTS...
        vseg_de_add_threshold = process.configuration.get_float("VIDEO_SEGMENTATION_DEL_EVENT_ADD_THRESHOLD", 10)
        vseg_de_min_segment_length = process.configuration.get_int("VIDEO_SEGMENTATION_DEL_EVENT_MIN_LENGTH", 15)
        vseg_de_threshold = process.configuration.get_float("VIDEO_SEGMENTATION_DEL_EVENT_THRESHOLD", 0.25)

        add_values = np.zeros(len(st3D.frame_indices))
        del_values = np.zeros(len(st3D.frame_indices))

        for group_idx in group_ages:
            first = group_ages[group_idx][0]
            last = group_ages[group_idx][-1]

            g_min_x, g_max_x, g_min_y, g_max_y = st3D.cc_group_boundaries[group_idx]

            area = (g_max_x - g_min_x + 1) * (g_max_y - g_min_y + 1)

            # normalize area by frame size ...
            area /= (st3D.width * st3D.height)

            add_values[first] += area
            del_values[last] += area

            # print((st3D.cc_group_boundaries[group_idx], first, last))

        # expanded = np.zeros((del_values.shape[0] + window_size - 1))
        # expanded[window_size - 1:] = del_values.copy()

        # view_expanded =  np.lib.stride_tricks.as_strided(expanded, shape=(del_values.shape[0], window_size),
        # strides = (expanded.strides[0], expanded.strides[0]))

        # right_accum_del = view_expanded.sum(axis=1)

        accumulated_delete = 0.0
        # accumulated_add = 0.0
        cumulative_delete = np.zeros(len(st3D.frame_indices))
        for idx in range(len(st3D.frame_indices)):
            if add_values[idx] > vseg_de_add_threshold:
                accumulated_delete = 0.0

            accumulated_delete += del_values[idx]
            cumulative_delete[idx] = accumulated_delete

        # get splits ...
        intervals = VideoSegmenter.split_video_from_group_deletes(cumulative_delete, 0, len(st3D.frame_indices) - 1,
                                                                  vseg_de_min_segment_length, vseg_de_threshold)

        # com_values = add_values + del_values
        print(intervals)
        print([(st3D.frame_indices[start_f], st3D.frame_indices[end_f]) for start_f, end_f in intervals])

        if debug_mode:
            plt.plot(st3D.frame_indices, del_values, label="Del Values")
            # plt.plot(st3D.frame_indices, com_values, label="Com Values")
            # plt.plot(st3D.frame_indices, right_accum_del, label="Right-acummulated Del Values (5)")
            plt.plot(st3D.frame_indices, add_values, label="Add values")
            plt.plot(st3D.frame_indices, cumulative_delete, label="Cumulative Delete")

            plt.legend()

            save_file_prefix = f"{process.img_dir}/del_event_signal_{process.current_lecture.title}_"
            plt.savefig(save_file_prefix, dpi=200)
            plt.close()

    elif segmentation_method == 2:
        # using conflict minimization ....
        group_ages, conflicts = input_data[1]

        # configuration overrides for conflict minimization...
        if "conf_w" in process.params:
            vseg_conf_weights = int(process.params["conf_w"])
        else:
            vseg_conf_weights = process.configuration.get_int("VIDEO_SEGMENTATION_CONFLICTS_WEIGHTS", 0)

        if "conf_p" in process.params:
            vseg_conf_weights_pixels = int(process.params["conf_p"])
        else:
            vseg_conf_weights_pixels = process.configuration.get_int("VIDEO_SEGMENTATION_CONFLICTS_WEIGHTS_PIXELS", 0)

        if "conf_t" in process.params:
            vseg_conf_weights_time = int(process.params["conf_t"])
        else:
            vseg_conf_weights_time = process.configuration.get_int("VIDEO_SEGMENTATION_CONFLICTS_WEIGHTS_TIME", 0)

        conf_min_conflicts = process.configuration.get("VIDEO_SEGMENTATION_CONFLICTS_MIN_CONFLICTS", 3.0)
        conf_min_split = process.configuration.get_int("VIDEO_SEGMENTATION_CONFLICTS_MIN_SPLIT", 50)
        conf_min_length = process.configuration.get_int("VIDEO_SEGMENTATION_CONFLICTS_MIN_LENGTH", 25)

        print((conf_min_conflicts, conf_min_split, conf_min_length))

        if vseg_conf_weights in [VideoSegmenter.ConflictsAreaWeightsIntersection,
                                 VideoSegmenter.ConflictsAreaWeigthsUnion]:
            h, w = all_binary[0].shape
            img_size = h * w

            for group_idx in conflicts:
                for other_idx in conflicts[group_idx]:
                    conflicts[group_idx][other_idx]["area_intersection"] /= img_size
                    conflicts[group_idx][other_idx]["area_union"] /= img_size

        # now, compute ideal intervals based on conflicts
        if debug_mode:
            save_file_prefix = f"{process.img_dir}/group_segment_{process.current_lecture.title}_"
        else:
            save_file_prefix = None

        intervals = VideoSegmenter.from_group_conflicts(len(frame_indices), group_ages, conflicts,
                                                        conf_min_conflicts, conf_min_split, conf_min_length,
                                                        vseg_conf_weights, vseg_conf_weights_pixels,
                                                        vseg_conf_weights_time, save_file_prefix)
    else:
        # using sums
        # minimum size of a leaf in the Regression Decision Tree
        sampling_fps = process.configuration.get_float("SAMPLING_FPS")
        sum_min_segment = process.configuration.get_int("VIDEO_SEGMENTATION_SUM_MIN_SEGMENT")
        sum_min_erase_ratio = process.configuration.get_float("VIDEO_SEGMENTATION_SUM_MIN_ERASE_RATIO")

        leaf_min = int(math.ceil(sum_min_segment * sampling_fps))

        intervals = VideoSegmenter.video_segments_from_sums(all_sums, leaf_min, sum_min_erase_ratio)
        print("Erasing Events: ")
        print(intervals)

    # 4) Debug output...
    if debug_mode:
        y = np.array(all_sums)

        # Plot the results
        fig = plt.figure(figsize=(8, 6), dpi=200)

        ax1 = fig.add_subplot(111)

        max_y_value = y.max() * 1.10

        X = np.arange(len(all_sums))
        ax1.fill_between(X, y, facecolor="#7777DD", alpha=0.5)

        if segmentation_method == 3:
            plt.title("Deletion Event Estimation Video Segmentation")
        elif segmentation_method == 2:
            plt.title("Conflict Minimization Video Segmentation")
        else:
            plt.title("Decision Tree Regression Video Segmentation")

            eval_X = np.arange(len(all_sums)).reshape(len(all_sums), 1)

            regressor = VideoSegmenter.create_regresor_from_sums(all_sums, leaf_min)
            y_1 = regressor.predict(eval_X)
            color = "#2222FF"
            plt.plot(eval_X, y_1, c=color, linewidth=2)
            # ax1.fill_between(X, y_1, c=color, linewidth=2, alpha=0.5)

        for start_idx, end_idx in intervals:
            first_x = X[start_idx]
            last_x = X[end_idx]

            plt.plot(np.array([first_x, first_x]), np.array([0, max_y_value]), c="g", linewidth=1)
            plt.plot(np.array([last_x, last_x]), np.array([0, max_y_value]), c="r", linewidth=1)

        plt.xlabel("data")
        plt.ylabel("target")

        # plt.legend()

        out_filename = f"{process.img_dir}/intervals_{segmentation_method}_{process.current_lecture.title}.png"
        plt.savefig(out_filename, dpi=200)

        plt.close()

    print("Total intervals: " + str(len(intervals)))

    return intervals


def main():
    # usage check
    if not ConsoleUIProcess.usage_with_config_check(sys.argv):
        return

    process = ConsoleUIProcess.FromConfigPath(sys.argv[1], sys.argv[2:], None, "VIDEO_SEGMENTATION_OUTPUT")

    # depending on segmentation method ....
    segmentation_method = process.configuration.get_int("VIDEO_SEGMENTATION_METHOD", 2)

    # check and chose inputs
    if segmentation_method == 3:
        # Deletion events (estimated from CC ST3D)
        inputs = [process.configuration.get("CC_RECONSTRUCTED_OUTPUT"),
                  process.configuration.get("CC_CONFLICTS_OUTPUT"),
                  process.configuration.get("CC_ST3D_OUTPUT")]
    elif segmentation_method == 2:
        # using CC conflicts to split the video
        inputs = [process.configuration.get("CC_RECONSTRUCTED_OUTPUT"),
                  process.configuration.get("CC_CONFLICTS_OUTPUT")]
    else:
        # using SUMS by default
        inputs = process.configuration.get("CC_RECONSTRUCTED_OUTPUT")

    # set inputs ....
    process.input_temp_prefix = inputs


    if not process.initialize():
        return

    start_time = time.time()
    process.start_input_processing(process_input)
    end_time = time.time()

    print("Total time: " + TimeHelper.secondsToStr(end_time - start_time))
    print("Finished")


if __name__ == '__main__':
    main()

