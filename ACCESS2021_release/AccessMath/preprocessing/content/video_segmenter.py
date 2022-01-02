
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

class VideoSegmenter:
    ConflictsAreaWeightsCount = 0
    ConflictsAreaWeigthsUnion = 3
    ConflictsAreaWeightsIntersection = 4
    ConflictsAreaWeightsIOU = 5

    ConflictsPixelsWeightsNone = 0
    ConflictsPixelsWeightsMatched = 1
    ConflictsPixelsWeightsUnmatched = 2
    ConflictsPixelsWeightsIOU = 3

    ConflictsTimeWeightNone = 0
    ConflictsTimeWeightGap = 1
    ConflictsTimeWeightNormalizedLength =2

    @staticmethod
    def compute_binary_sums(all_binary):
        all_sums = []

        for binary in all_binary:
            all_sums.append(binary.sum() / 255)

        return all_sums

    @staticmethod
    def create_regresor_from_sums(all_sums, leaf_min):
        # capture values ...
        train_data_x = np.arange(0, len(all_sums), dtype=np.int32).reshape(len(all_sums), 1)
        train_data_y = np.array(all_sums)

        # Fit regression models
        tree = DecisionTreeRegressor(max_depth=None, min_samples_leaf=leaf_min)
        tree.fit(train_data_x, train_data_y)

        return tree

    @staticmethod
    def get_tree_decision_boundaries(regressor_tree, max_x):
        X = np.arange(0, max_x, dtype=np.int32).reshape(max_x, 1)
        y_1 = regressor_tree.predict(X)

        # find decision boundaries ...
        interval_idxs = [0]
        interval_vals = [y_1[0]]
        for idx in range(1, X.shape[0]):
            if y_1[idx] != interval_vals[-1]:
                interval_idxs.append(idx)
                interval_vals.append(y_1[idx])

        return interval_idxs, interval_vals

    @staticmethod
    def identify_descend_intervals(interval_vals, min_pixels_erased):
        # now identify continuous descent regions
        descend_intervals = []
        current_start = None
        for idx in range(1, len(interval_vals)):
            if interval_vals[idx] < interval_vals[idx -1]:
                if current_start is None:
                    current_start = idx
            else:
                if current_start is not None:
                    descend_intervals.append((current_start, idx - 1))
                    current_start = None

        # close last interval (if it was descend)
        if current_start is not None:
            descend_intervals.append((current_start, len(interval_vals) - 1))

        # filter descents which really don't affect the content
        filtered_intervals = []
        for start_idx, end_idx in descend_intervals:
            diff = interval_vals[start_idx - 1] - interval_vals[end_idx]
            ratio = interval_vals[end_idx] / interval_vals[start_idx - 1]

            #if ratio < 1.0 - min_erase_ratio:
            if diff >= min_pixels_erased:
                #print("Ratio : " + str(ratio) + ", Diff: " + str(interval_vals[start_idx - 1] - interval_vals[end_idx]) + ", min = " + str(min_pixels_erased))
                filtered_intervals.append( (start_idx, end_idx))

        return filtered_intervals

    @staticmethod
    def video_segments_from_erasing_intervals(erasing_intervals, n_images):
        video_segments = []
        current_start = 0
        for start_erase, end_erase in erasing_intervals:
            video_segments.append((current_start, start_erase -1))
            current_start = end_erase + 1

        if current_start < n_images - 1:
            video_segments.append((current_start, n_images - 1))

        return video_segments

    @staticmethod
    def video_segments_from_sums(all_sums, min_points, min_erase):
        # get the average number of pixels on the board
        sum_array = np.array(all_sums)
        avg_sum = sum_array.mean()

        min_pixels_erased = avg_sum * min_erase

        # Fit a Regression Decision Tree
        regressor = VideoSegmenter.create_regresor_from_sums(all_sums, min_points)

        # Find decision boundaries
        interval_idxs, interval_vals = VideoSegmenter.get_tree_decision_boundaries(regressor, len(all_sums))

        descending_intervals = VideoSegmenter.identify_descend_intervals(interval_vals, min_pixels_erased)

        # refine boundaries for intervals ...
        refined_intervals = []
        for start_idx, end_idx in descending_intervals:
            # find maximum in erase region ...
            if end_idx + 1 < len(interval_idxs):
                last_x = interval_idxs[end_idx + 1]
            else:
                last_x = len(all_sums) - 1

            refined_intervals.append((interval_idxs[start_idx], last_x))

        segments = VideoSegmenter.video_segments_from_erasing_intervals(refined_intervals, len(all_sums))

        return segments

    @staticmethod
    def find_signal_peaks(start_frame, end_frame, signal_dict):
        all_peaks = []
        current_peak_start = None
        current_peak_highest = None
        going_up = None
        for frame_idx in range(start_frame, end_frame + 1):
            if current_peak_start is None:
                # starting a new peak region ...
                current_peak_start = frame_idx
                current_peak_highest = frame_idx
                # assume we start going up ...
                going_up = True
            else:
                # there are three posibilities:
                # 1) the value is higher than previous, check status
                #    1.a) going up? peak is just growing, update it!
                #    1.b) going down? last peak ends and new one starts!
                # 2) the value is lower than previous. no longer going up!
                # 3) the value is the same. A plateu is extending, nothing to do here!
                if signal_dict[frame_idx] > signal_dict[frame_idx - 1]:
                    # check if we were going up ...
                    if going_up:
                        # still going up ... update peak
                        current_peak_highest = frame_idx
                    else:
                        # we were not going up but now we start going up again, new peak starts!
                        all_peaks.append((current_peak_start, current_peak_highest, frame_idx -1))
                        # restart peak search ...
                        current_peak_start = frame_idx
                        current_peak_highest = frame_idx
                        # assume we start going up ...
                        going_up = True
                else:
                    # check ....
                    if signal_dict[frame_idx] < signal_dict[frame_idx - 1]:
                        # signal is lower ... check
                        if going_up:
                            # mark that we start going down!
                            going_up = False
                        else:
                            # we were going down already, nothing to do for now
                            pass
                    else:
                        # a plateu (either going up or down ... nothing to do for now)
                        pass

        if current_peak_start is not None:
            # save last peak
            all_peaks.append((current_peak_start, current_peak_highest, end_frame))

        return all_peaks

    @staticmethod
    def split_video_from_group_conflicts(start_frame, end_frame, group_ages, group_conflicts, min_conflicts,
                                         min_segment_split, min_segment_len,
                                         method_weight_area, method_weight_pixels, method_weight_time,
                                         current_depth, graph_data, split_data, n_frames):

        # check base case (min segment length)
        if end_frame - start_frame + 1 < min_segment_split:
            print(str([(start_frame, end_frame)]) + " cannot split, too small")
            return [(start_frame, end_frame)]

        # check which groups exists in the current range
        current_groups = []
        for group_idx in group_ages:
            first = group_ages[group_idx][0]
            last = group_ages[group_idx][-1]
            if start_frame <= last and first <= end_frame:
                current_groups.append(group_idx)

        #print("groups from " + str(start_frame) + " to " + str(end_frame) + " = " + str(len(current_groups)))

        # count conflicts per frame
        conflicts_per_frame = {x: 0.0 for x in range(start_frame, end_frame + 1)}
        # for each of the current groups ...
        for group_idx in current_groups:
            group_first = group_ages[group_idx][0]
            group_last = group_ages[group_idx][-1]

            # check conflicts
            for other_idx in group_conflicts[group_idx]:
                # only check conflicts once and only consider groups in the current section
                if group_idx < other_idx and other_idx in current_groups:
                    # count the conflict in the intermediate frames ...
                    other_first = group_ages[other_idx][0]
                    other_last = group_ages[other_idx][-1]

                    # check which group is older ...
                    if group_first < other_first:
                        # current group is older
                        # conflict_start = group_last + 1
                        conflict_start = group_last
                        conflict_end = other_first - 1
                    else:
                        # other group is older
                        # conflict_start = other_last + 1
                        conflict_start = other_last
                        conflict_end = group_first - 1

                    #print(str(group_idx) + "\t" + str(other_idx) + "\t" + str(conflict_start) + "\t" + str(conflict_end))

                    # temporal weighting
                    if method_weight_time == VideoSegmenter.ConflictsTimeWeightGap:
                        time_weight = (conflict_end - conflict_start + 1)
                    elif method_weight_time == VideoSegmenter.ConflictsTimeWeightNormalizedLength:
                        length_group = group_last - group_first + 1
                        length_other = other_last - other_first + 1
                        time_weight = (length_group + length_other) / n_frames
                    else:
                        # default .... all conflicts weight the same on time
                        time_weight = 1

                    if method_weight_pixels == VideoSegmenter.ConflictsPixelsWeightsMatched:
                        pixel_weight = group_conflicts[group_idx][other_idx]["matched"]
                    elif method_weight_pixels == VideoSegmenter.ConflictsPixelsWeightsUnmatched:
                        pixel_weight = group_conflicts[group_idx][other_idx]["unmatched"]
                    elif method_weight_pixels == VideoSegmenter.ConflictsPixelsWeightsIOU:
                        intersection = group_conflicts[group_idx][other_idx]["matched"]
                        union = (group_conflicts[group_idx][other_idx]["matched"] +
                                 group_conflicts[group_idx][other_idx]["unmatched"])

                        # inverse of IOU (the less they intersect, the higher the conflict weight)
                        pixel_weight = 1 - (intersection / union)
                    else:
                        # default ... all conflicts weight the same on pixels
                        pixel_weight = 1

                    # Spatial weighting
                    if method_weight_area == VideoSegmenter.ConflictsAreaWeightsIntersection:
                        area_weight = group_conflicts[group_idx][other_idx]["area_intersection"]
                    elif method_weight_area == VideoSegmenter.ConflictsAreaWeigthsUnion:
                        area_weight = group_conflicts[group_idx][other_idx]["area_union"]
                    elif method_weight_area == VideoSegmenter.ConflictsAreaWeightsIOU:
                        iou = (group_conflicts[group_idx][other_idx]["area_intersection"] /
                               group_conflicts[group_idx][other_idx]["area_union"])
                        area_weight = iou
                    else:
                        # default ... weight by simple conflict count (all conflicts are equal!)
                        area_weight = 1

                    conflict_weight = area_weight * pixel_weight * time_weight

                    # add ...
                    for frame_idx in range(conflict_start, conflict_end + 1):
                        conflicts_per_frame[frame_idx] += conflict_weight

                    """
                    if conflict_end - conflict_start > -10:
                        print("\n\n\nHere!")
                        print((group_first, group_last, other_first, other_last, conflict_start, conflict_end))
                        print((conflict_weight, area_weight, pixel_weight, time_weight))
                        print("\n\n\n")
                    """


        graph_data.append((current_depth, conflicts_per_frame))

        peaks_of_conflict = VideoSegmenter.find_signal_peaks(start_frame, end_frame, conflicts_per_frame)
        # print(peaks_of_conflict)
        # print(len(conflicts_per_frame))
        # print(len(peaks_of_conflict))

        """
        # find split frame
        best_split = start_frame
        for frame_idx in range(start_frame, end_frame + 1):
            #print(str(frame_idx) + "\t" + str(conflicts_per_frame[frame_idx]))
            if conflicts_per_frame[frame_idx] > conflicts_per_frame[best_split]:
                best_split = frame_idx
        """

        """
        candidate_splits = []
        too_short = True
        for frame_idx in range(start_frame + min_segment_len, end_frame - min_segment_len + 1):
            too_short = False
            if conflicts_per_frame[frame_idx] > min_conflicts:
                candidate_splits.append((conflicts_per_frame[frame_idx], frame_idx))

        candidate_splits = sorted(candidate_splits, reverse=True)
        """

        candidate_peaks = []
        for peak_start, peak_highest, peak_end in peaks_of_conflict:
            if ((conflicts_per_frame[peak_highest] > min_conflicts) and
                (start_frame + min_segment_len <= peak_highest <= end_frame - min_segment_len)):
                candidate_peaks.append((conflicts_per_frame[peak_highest], peak_highest))

        candidate_peaks = sorted(candidate_peaks, reverse=True)

        """
        if len(candidate_splits) > 0:
            print(candidate_splits[0:5])
            print(candidate_peaks[0:5])

            if len(candidate_peaks) == 0:
                print("CASO C!")
                print(peaks_of_conflict)
                print((start_frame, end_frame))
                x = 0 / 0

            if abs(candidate_peaks[0][0] - candidate_splits[0][0]) > 30:
                print(peaks_of_conflict)
                print((start_frame, end_frame))
                x = 0 / 0
        else:
            if len(candidate_peaks) > 0:
                print("case B")
                x = 0 / 0
        """

        """
        # if conflicts_per_frame[best_split] < min_conflicts:
        if len(candidate_splits) == 0:
            # base case, not enough conflicts to split ...
            if weight_method == VideoSegmenter.ConflictsWeightsCount:
                conf_name = "conflicts"
            elif weight_method == VideoSegmenter.ConflictsWeightsMatchedPixels:
                conf_name = "matched pixels"
            elif weight_method == VideoSegmenter.ConflictsWeightsUnmatchedPixels:
                conf_name = "unmatched pixels"
            else:
                conf_name = "<???>"

            if too_short:
                print(str([(start_frame, end_frame)]) + " too small segments")
            else:
                print(str([(start_frame, end_frame)]) + " not enough " + conf_name)

            return [(start_frame, end_frame)]

        conflict_count, best_split = candidate_splits[0]
        """

        if len(candidate_peaks) == 0:
            print(str([(start_frame, end_frame)]) + " no good split candidates found")

            return [(start_frame, end_frame)]

        # conflict_count, best_split = candidate_splits[0]
        conflict_count, best_split = candidate_peaks[0]

        """
        print("Current split: " + str(start_frame) + " - " + str(end_frame) + ", frame " +
              str(best_split) + " (" + str(conflicts_per_frame[best_split]) + " conflicts)" )
        """
        # if (best_split - start_frame >= min_segment_len) and (end_frame - best_split >= min_segment_len):

        split_data.append((current_depth, best_split))

        # split parts recursively
        left = VideoSegmenter.split_video_from_group_conflicts(start_frame, best_split - 1, group_ages,
                                                               group_conflicts, min_conflicts,
                                                               min_segment_split, min_segment_len,
                                                               method_weight_area, method_weight_pixels,
                                                               method_weight_time, current_depth + 1,
                                                               graph_data, split_data, n_frames)
        right = VideoSegmenter.split_video_from_group_conflicts(best_split + 1, end_frame, group_ages,
                                                                group_conflicts, min_conflicts,
                                                                min_segment_split, min_segment_len,
                                                                method_weight_area, method_weight_pixels,
                                                                method_weight_time, current_depth + 1,
                                                                graph_data, split_data, n_frames)

        return left + right

    @staticmethod
    def merge_conflict_plot_data(graph_data, n_frames):
        # assumes a list of tuples (depth, data), must generate a list of arrays

        # first, detect recursion depth ...
        max_depth = 0
        for depth, data in graph_data:
            if depth > max_depth:
                max_depth = depth

        # now, create empty arrays per level of depth
        final_arrays = []
        for depth in range(max_depth + 1):
            final_arrays.append(np.zeros(n_frames, dtype=np.float32))

        # fill the arrays ...
        for depth, data in graph_data:
            depth_array = final_arrays[depth]

            for frame_idx in data:
                depth_array[frame_idx] = data[frame_idx]

        return final_arrays

    @staticmethod
    def save_conflict_plot(n_frames, graph_data, split_data, filename, min_depth=0):
        colors_areas = ["#7777DD", "#77DD77", "#DD7777", "#DDDD77", "#77DDDD"]
        colors_splits = ["#222288", "#228822", "#882222", "#888822", "#228888"]

        if min_depth >= len(graph_data):
            print("WARNING: Cannot generate conflict plot at Depth <" + str(min_depth))
            return

        fig = plt.figure()

        ax1 = fig.add_subplot(111)

        x = np.arange(n_frames)

        # add the filled elements
        for depth, depth_array in enumerate(graph_data):
            if depth < min_depth:
                continue
            ax1.fill_between(x, depth_array, facecolor=colors_areas[depth % len(colors_areas)])

        # add the vertical lines
        max_y_value = graph_data[min_depth].max()
        for depth, split_x in split_data:

            #plt
            ax1.plot(np.array([split_x, split_x]), np.array([0, max_y_value]), c=colors_splits[depth % len(colors_splits)], linewidth=1)

        plt.savefig(filename,dpi=200)
        plt.close()


    @staticmethod
    def from_group_conflicts(n_frames, group_ages, group_conflicts, min_conflicts, min_split, min_len,
                             weight_area, weight_pixels, weight_time, save_prefix=None):
        # generate segments and collect graph data ...
        graph_data = []
        split_data = []
        segments = VideoSegmenter.split_video_from_group_conflicts(0, n_frames - 1, group_ages, group_conflicts,
                                                                   min_conflicts, min_split, min_len, weight_area,
                                                                   weight_pixels, weight_time, 0, graph_data,
                                                                   split_data, n_frames)

        if save_prefix is not None:
            graph_data = VideoSegmenter.merge_conflict_plot_data(graph_data, n_frames)
            for depth in range(3):
                plot_name = save_prefix + "plot_depth_" + str(depth) + ".png"
                VideoSegmenter.save_conflict_plot(n_frames, graph_data, split_data, plot_name, depth)

        return segments

    @staticmethod
    def from_group_conflicts_with_presegments(n_frames, pre_segments, group_ages, group_conflicts, min_conflicts,
                                              min_split, min_len, weight_area, weight_pixels, weight_time,
                                              save_prefix=None):
        graph_data = []
        split_data = []

        all_segments = []
        for seg_start, seg_end in pre_segments:
            segments = VideoSegmenter.split_video_from_group_conflicts(seg_start, seg_end, group_ages, group_conflicts,
                                                                       min_conflicts, min_split, min_len, weight_area,
                                                                       weight_pixels, weight_time, 0, graph_data,
                                                                       split_data, n_frames)
            all_segments += segments

        if save_prefix is not None:
            graph_data = VideoSegmenter.merge_conflict_plot_data(graph_data, n_frames)
            for depth in range(3):
                plot_name = save_prefix + "plot_depth_" + str(depth) + ".png"
                VideoSegmenter.save_conflict_plot(n_frames, graph_data, split_data, plot_name, depth)

        return all_segments

    @staticmethod
    def split_video_from_group_deletes(signal, start_frame, end_frame, min_length, threshold):
        peaks_of_delete = VideoSegmenter.find_signal_peaks(start_frame, end_frame, signal)

        candidate_peaks = []
        for peak_start, peak_highest, peak_end in peaks_of_delete:
            if ((signal[peak_highest] > threshold) and
                (start_frame + min_length <= peak_highest <= end_frame - min_length)):
                candidate_peaks.append((signal[peak_highest], peak_highest))

        candidate_peaks = sorted(candidate_peaks, reverse=True)

        if len(candidate_peaks) == 0:
            print(str([(start_frame, end_frame)]) + " no good split candidates found")
            return [(start_frame, end_frame)]

        signal_value, best_split = candidate_peaks[0]

        left = VideoSegmenter.split_video_from_group_deletes(signal, start_frame, best_split - 1, min_length, threshold)
        right = VideoSegmenter.split_video_from_group_deletes(signal, best_split + 1, end_frame, min_length, threshold)

        return left + right

