
import sys
import json

import numpy as np

from AM_CommonTools.configuration.configuration import Configuration

from AccessMath.data.meta_data_DB import MetaDataDB
from AccessMath.annotation.lecture_annotation import LectureAnnotation
from AccessMath.util.misc_helper import MiscHelper


def get_overlaps(gt_segments, abs_pred_segments):
    gt_pos = 0
    pred_pos = 0
    overlaps = []
    while gt_pos < len(gt_segments) - 1 and pred_pos < len(abs_pred_segments):
        gt_segment_start = gt_segments[gt_pos]
        gt_segment_end = gt_segments[gt_pos + 1]

        pred_segment_start, pred_segment_end = abs_pred_segments[pred_pos]

        # check if these overlap
        if gt_segment_start < pred_segment_end and pred_segment_start < gt_segment_end:
            # compute IOU
            intersection = min(gt_segment_end, pred_segment_end) - max(gt_segment_start, pred_segment_start)
            union = max(gt_segment_end, pred_segment_end) - min(gt_segment_start, pred_segment_start)

            IOU = intersection / union
            overlaps.append((IOU, gt_pos, pred_pos))

        # move the one that finishes first!!!
        if gt_segment_end < pred_segment_end:
            gt_pos += 1
        else:
            pred_pos += 1

    return overlaps


def from_segments_to_split_points(segments):
    split_points = []
    for idx in range(len(segments) - 1):
        pre_start, pre_end = segments[idx]
        next_start, next_end = segments[idx + 1]

        split = int((pre_end + next_start) / 2)
        split_points.append(split)

    return split_points


def get_average_segment_length(segments):
    all_lengths = []
    for start, end in segments:
        length = end - start
        all_lengths.append(length)

    avg_segment_length = sum(all_lengths) / len(all_lengths)

    return avg_segment_length


def match_split_points(gt_split_points, pred_split_points, max_match_gap):
    all_pairs = []
    for gt_split in gt_split_points:
        for pred_split in pred_split_points:
            dist = abs(gt_split - pred_split)

            all_pairs.append((dist, gt_split, pred_split))

    all_pairs = sorted(all_pairs)

    gt_matched = {}
    pred_matched = {}
    matches = []
    for dist, gt_split, pred_split in all_pairs:
        if dist < max_match_gap:
            if gt_split not in gt_matched and pred_split not in pred_matched:
                gt_matched[gt_split] = True
                pred_matched[pred_split] = True

                matches.append((gt_split, pred_split))
        else:
            # no more matches can be found after this point
            break

    return matches


def show_summary(results_per_lecture, sizes_gt, sizes_pred, group_name):
    print("\nPer Lecture summary ({0:s})".format(group_name))
    print("Lecture\tGT\tPred.\tMatches\tAvg. IOU")
    avg_gt_count = 0
    avg_pred_count = 0
    avg_match_count = 0.0
    avg_SIOU = 0.0
    for lecture_title, count_gt, count_pred, count_match, avg_IOU in results_per_lecture:
        avg_gt_count += count_gt
        avg_pred_count += count_pred
        avg_match_count += (count_match / count_gt)
        avg_SIOU += avg_IOU

        print("{0:s}\t{1:d}\t{2:d}\t{3:d}\t{4:.4f}".format(lecture_title, count_gt, count_pred, count_match, avg_IOU))

    avg_gt_count /= len(results_per_lecture)
    avg_pred_count /= len(results_per_lecture)
    avg_match_count /= len(results_per_lecture)
    avg_SIOU /= len(results_per_lecture)

    print("AVG\t{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}".format(avg_gt_count, avg_pred_count, avg_match_count, avg_SIOU))

    print("\n{0:s} - Average Segments on GT: {1:.2f}".format(group_name, np.mean(sizes_gt)))
    print("{0:s} - Average Segments on Pred: {1:.2f}".format(group_name, np.mean(sizes_pred)))


def show_segment_stats(stats_title, stats_level, keys_name, range_keys, stats_per_range):
    print("\n\n{0:s} ({1:s})".format(stats_title, stats_level))
    print("{0:s}\tRec.\tPrec.\tF-1".format(keys_name))
    for range_key in range_keys:
        mean_recall = np.mean(stats_per_range[range_key]["recalls"])
        mean_precision = np.mean(stats_per_range[range_key]["precisions"])

        if mean_recall + mean_precision > 0.0:
            f1 = (2.0 * mean_recall * mean_precision) / (mean_recall + mean_precision)
        else:
            f1 = 0.0

        print("{0}\t{1:.2f}\t{2:.2f}\t{3:.2f}".format(range_key, mean_recall * 100.0,
                                                      mean_precision * 100.0, f1 * 100.0))

def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("\tpython {0:s} config training [edited_gt]".format(sys.argv[0]))
        print("Where")
        print("\tconfig:\t\tPath to Configuration File")
        print("\ttraining:\t 1 for training set, 0 for testing set")
        print("\tedited_gt:\tOptional, allows to split metrics between edited and non-edited videos")
        print("\t\tA dictionary in JSON file is expected, where each lecture is represented by title and ")
        print("\t\tthe corresponding per-lecture dictionary contains the boolean edited field")
        return

    # read the configuration file ....
    config = Configuration.from_file(sys.argv[1])

    try:
        database = MetaDataDB.from_file(config.get_str("VIDEO_DATABASE_PATH"))
    except:
        print("Invalid AccessMath Database file")
        return

    try:
        use_training = int(sys.argv[2]) > 0
    except:
        print("Invalid value for parameter: training")
        return

    if len(sys.argv) >= 4:
        edited_filename = sys.argv[3]
        with open(edited_filename, "r") as in_file:
            edited_gt = json.load(in_file)
    else:
        edited_gt = None

    output_dir = config.get_str("OUTPUT_PATH")
    temporal_segments_dir = output_dir + "/" + database.output_temporal

    binary_prefix = config.get_str("BINARIZATION_OUTPUT")
    segments_prefix = config.get_str("VIDEO_SEGMENTATION_OUTPUT")

    if use_training:
        dataset_name = "training"
    else:
        dataset_name = "testing"

    current_dataset = database.datasets[dataset_name]
    target_IOU_levels = [0.5, 0.6, 0.7, 0.8, 0.9]
    target_max_gaps = [90, 150, 210, 300]
    target_prop_gaps = [0.025, 0.05, 0.075, 0.100]

    metrics_per_IOU_all = {}
    metrics_per_IOU_edited = {}
    metrics_per_IOU_non_edited = {}
    for iou in target_IOU_levels:
        metrics_per_IOU_all[iou] = {"recalls": [], "precisions": []}
        metrics_per_IOU_edited[iou] = {"recalls": [], "precisions": []}
        metrics_per_IOU_non_edited[iou] = {"recalls": [], "precisions": []}

    metrics_per_GAP_all = {}
    metrics_per_GAP_edited = {}
    metrics_per_GAP_non_edited = {}
    for max_gap in target_max_gaps:
        metrics_per_GAP_all[max_gap] = {"recalls": [], "precisions": []}
        metrics_per_GAP_edited[max_gap] = {"recalls": [], "precisions": []}
        metrics_per_GAP_non_edited[max_gap] = {"recalls": [], "precisions": []}

    metrics_per_PROP_all = {}
    metrics_per_PROP_edited = {}
    metrics_per_PROP_non_edited = {}
    for prop_gap in target_prop_gaps:
        metrics_per_PROP_all[prop_gap] = {"recalls": [], "precisions": []}
        metrics_per_PROP_edited[prop_gap] = {"recalls": [], "precisions": []}
        metrics_per_PROP_non_edited[prop_gap] = {"recalls": [], "precisions": []}

    sizes_gt = []
    sizes_pred = []
    edited_sizes_gt = []
    edited_sizes_pred = []
    non_edited_sizes_gt = []
    non_edited_sizes_pred = []
    results_per_lecture = []
    results_per_edited_lecture = []
    results_per_non_edited_lecture = []

    for current_lecture in current_dataset:
        print("Processing: " + current_lecture.title)
        if edited_gt is not None:
            if edited_gt[current_lecture.title]["edited"]:
                print("-> Edited Lecture Video")
            else:
                print("-> Non-edited Lecture Video")


        # read segment data ....
        input_filename = temporal_segments_dir + "/" + segments_prefix + current_lecture.title + ".dat"
        video_segment_data = MiscHelper.dump_load(input_filename)

        # read binary data ...
        input_filename = temporal_segments_dir + "/" + binary_prefix + current_lecture.title + ".dat"
        binary_data = MiscHelper.dump_load(input_filename)
        frame_times, frame_indices, compressed_frames = binary_data

        # read ground truth for this lecture ...
        input_filename = (output_dir + "/" + database.output_annotations + "/" +
                          database.name + "_" + current_lecture.title.lower() + ".xml")
        annotation = LectureAnnotation.Load(input_filename, False)

        gt_segments = [0] + annotation.video_segments + [annotation.total_frames]

        # convert video segments from sample offsets to absolute frame ids
        abs_pred_segments = []
        for start_offset, end_offset in video_segment_data:
            start_idx = frame_indices[start_offset]
            end_idx = frame_indices[end_offset]

            abs_pred_segments.append((start_idx, end_idx))

        sizes_gt.append(len(gt_segments) - 1)
        sizes_pred.append(len(abs_pred_segments))

        if edited_gt is not None:
            if edited_gt[current_lecture.title]["edited"]:
                edited_sizes_gt.append(len(gt_segments) - 1)
                edited_sizes_pred.append(len(abs_pred_segments))
            else:
                non_edited_sizes_gt.append(len(gt_segments) - 1)
                non_edited_sizes_pred.append(len(abs_pred_segments))

        print("\nTotal GT Segments: {0:d}".format(len(gt_segments) - 1))
        print("Total Pred. Segments: {0:d}".format(len(abs_pred_segments)))

        # find overlapping the segments
        overlaps = get_overlaps(gt_segments, abs_pred_segments)

        overlaps = sorted(overlaps, reverse=True)

        # 1) compute the 1-to-1 matching metrics
        print("IOU\tMatch\tRec.\tPrec.\tF-1")
        current_lecture_info = []
        for iou_idx, iou in enumerate(target_IOU_levels):
            matched_gt = {}
            matched_pred = {}
            count_matches = 0
            for match_iou, gt_pos, pred_pos in overlaps:
                if match_iou >= iou and (not gt_pos in matched_gt) and (not pred_pos in matched_pred):
                    matched_gt[gt_pos] = True
                    matched_pred[pred_pos] = True
                    count_matches += 1

            recall = count_matches / (len(gt_segments) - 1)
            precision = count_matches / len(abs_pred_segments)

            if recall + precision > 0.0:
                f1 = (2.0 * recall * precision) / (recall + precision)
            else:
                f1 = 0.0

            metrics_per_IOU_all[iou]["recalls"].append(recall)
            metrics_per_IOU_all[iou]["precisions"].append(precision)
            if edited_gt is not None:
                if edited_gt[current_lecture.title]["edited"]:
                    metrics_per_IOU_edited[iou]["recalls"].append(recall)
                    metrics_per_IOU_edited[iou]["precisions"].append(precision)
                else:
                    metrics_per_IOU_non_edited[iou]["recalls"].append(recall)
                    metrics_per_IOU_non_edited[iou]["precisions"].append(precision)

            print("{0:.2f}\t{1:d}\t{2:.2f}\t{3:.2f}\t{4:.2f}".format(iou, count_matches, recall * 100.0,
                                                                     precision * 100.0, f1 * 100.0))

            if iou_idx == 0:
                current_lecture_info += [current_lecture.title, len(gt_segments) - 1, len(abs_pred_segments),
                                         count_matches]

        # 2) compute SIoU (best IOU per GT region)
        overlaps = sorted([(gt_pos, match_iou, pred_pos) for match_iou, gt_pos, pred_pos in overlaps], reverse=True)

        # choose only top match for each GT segment
        current_gt = None
        matching_IOUs = []
        for gt_pos, match_iou, pred_pos in overlaps:
            if current_gt != gt_pos:
                # found the first match (best match) for this GT segment
                # print((gt_pos, pred_pos, match_iou))
                matching_IOUs.append(match_iou)
                current_gt = gt_pos

        # print(overlaps)
        avg_best_matching_IOU = sum(matching_IOUs) / len(matching_IOUs)
        current_lecture_info.append(avg_best_matching_IOU)

        # 3) Compute Split-based matches
        gt_split_points = annotation.video_segments
        pred_split_points = from_segments_to_split_points(abs_pred_segments)

        print("\nGT split points")
        print(gt_split_points)
        print("Predicted split points")
        print(pred_split_points)

        print("\nMx Gap\tMatch\tRec.\tPrec.\tF-1")
        table_row = "{0:d}\t{1:d}\t{2:.2f}\t{3:.2f}\t{4:.2f}"
        for max_match_gap in target_max_gaps:
            split_matches = match_split_points(gt_split_points, pred_split_points, max_match_gap)
            # split points recall
            if len(gt_split_points) > 0:
                recall = len(split_matches) / len(gt_split_points)
            else:
                recall = 1.0
            # split points precision
            if len(pred_split_points) > 0:
                precision = len(split_matches) / len(pred_split_points)
            else:
                """
                if len(gt_split_points) == 0:
                    precision = 1.0
                else:
                    precision = 0.0
                """
                precision = 1.0

            if precision + recall > 0.0:
                f_measure = (2 * precision * recall) / (precision + recall)
            else:
                f_measure = 0.0

            metrics_per_GAP_all[max_match_gap]["recalls"].append(recall)
            metrics_per_GAP_all[max_match_gap]["precisions"].append(precision)
            if edited_gt is not None:
                if edited_gt[current_lecture.title]["edited"]:
                    metrics_per_GAP_edited[max_match_gap]["recalls"].append(recall)
                    metrics_per_GAP_edited[max_match_gap]["precisions"].append(precision)
                else:
                    metrics_per_GAP_non_edited[max_match_gap]["recalls"].append(recall)
                    metrics_per_GAP_non_edited[max_match_gap]["precisions"].append(precision)

            print(table_row.format(max_match_gap, len(split_matches), recall, precision, f_measure))

        # 4) split matching with proportional length
        avg_segment_length = annotation.total_frames / (len(gt_segments) - 1)
        # print(avg_segment_length)
        # print(avg_segment_length * 0.05)
        # print(avg_segment_length * 0.05 / 29.97)

        print("\nGap Pr.\tMx Gap\tMatch\tRec.\tPrec.\tF-1")
        table_row = "{0:.4f}\t({1:.2f})\t{2:d}\t{3:.2f}\t{4:.2f}\t{5:.2f}"
        for prop_gap in target_prop_gaps:
            max_match_gap = avg_segment_length * prop_gap
            split_matches = match_split_points(gt_split_points, pred_split_points, max_match_gap)
            # split points recall
            if len(gt_split_points) > 0:
                recall = len(split_matches) / len(gt_split_points)
            else:
                recall = 1.0
            # split points precision
            if len(pred_split_points) > 0:
                precision = len(split_matches) / len(pred_split_points)
            else:
                precision = 1.0
                """
                if len(gt_split_points) == 0:
                    precision = 1.0
                else:
                    precision = 0.0
                """

            if precision + recall > 0.0:
                f_measure = (2 * precision * recall) / (precision + recall)
            else:
                f_measure = 0.0

            metrics_per_PROP_all[prop_gap]["recalls"].append(recall)
            metrics_per_PROP_all[prop_gap]["precisions"].append(precision)
            if edited_gt is not None:
                if edited_gt[current_lecture.title]["edited"]:
                    metrics_per_PROP_edited[prop_gap]["recalls"].append(recall)
                    metrics_per_PROP_edited[prop_gap]["precisions"].append(precision)
                else:
                    metrics_per_PROP_non_edited[prop_gap]["recalls"].append(recall)
                    metrics_per_PROP_non_edited[prop_gap]["precisions"].append(precision)

            print(table_row.format(prop_gap, max_match_gap, len(split_matches), recall, precision, f_measure))

        # add for later summary printing
        results_per_lecture.append(current_lecture_info)
        if edited_gt is not None:
            if edited_gt[current_lecture.title]["edited"]:
                results_per_edited_lecture.append(current_lecture_info)
            else:
                results_per_non_edited_lecture.append(current_lecture_info)

        print("\n")

    # show all summarized metrics for segment prediction
    if edited_gt is not None:
        show_summary(results_per_non_edited_lecture, non_edited_sizes_gt, non_edited_sizes_pred, "Non-Edited")
        show_summary(results_per_edited_lecture, edited_sizes_gt, edited_sizes_pred, "Edited")

    show_summary(results_per_lecture, sizes_gt, sizes_pred, "All Lectures")

    # Show metrics for matching segments at increasing IOU levels
    if edited_gt is not None:
        show_segment_stats("Segment Matching by IOU Level", "Non-edited", "IOU", target_IOU_levels, metrics_per_IOU_non_edited)
        show_segment_stats("Segment Matching by IOU Level", "Edited", "IOU", target_IOU_levels, metrics_per_IOU_edited)

    show_segment_stats("Segment Matching by IOU Level", "All Lectures", "IOU", target_IOU_levels, metrics_per_IOU_all)

    # Show metrics for matching split points at increasing time intervals (abs intervals)
    if edited_gt is not None:
        show_segment_stats("Split Matching by Max-Gap Level", "Non-edited", "Mx_Gap", target_max_gaps, metrics_per_GAP_non_edited)
        show_segment_stats("Split Matching by Max-Gap Level", "Edited", "Mx_Gap", target_max_gaps, metrics_per_GAP_edited)

    show_segment_stats("Split Matching by Max-Gap Level", "All Lectures", "Mx_Gap", target_max_gaps, metrics_per_GAP_all)

    # Show metrics for matching split points at increasing time intervals (proportional intervals)
    if edited_gt is not None:
        show_segment_stats("Split Matching by Max-Gap Proportional to each AVG Segment Length", "Non-edited",
                           "Pr_Gap", target_prop_gaps, metrics_per_PROP_non_edited)
        show_segment_stats("Split Matching by Max-Gap Proportional to each AVG Segment Length", "Edited",
                           "Pr_Gap", target_prop_gaps, metrics_per_PROP_edited)
    show_segment_stats("Split Matching by Max-Gap Proportional to each AVG Segment Length", "All Lectures",
                       "Pr_Gap", target_prop_gaps, metrics_per_PROP_all)


if __name__ == "__main__":
    main()

