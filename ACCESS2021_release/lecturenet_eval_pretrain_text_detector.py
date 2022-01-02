
import sys

from PIL import Image, ImageOps

import cv2
import numpy as np
from munkres import Munkres

import torch

from AM_CommonTools.configuration.configuration import Configuration

from AccessMath.lecturenet_v1.FCN_lecturenet import FCN_LectureNet
from AccessMath.lecturenet_v1.util import LectureNet_Util


def compute_matching(out_binary, gt_binary, IOU_tresholds, get_visualization=False):
    # 1) label CCs and get their boundaries on binary image
    out_n_labels, out_labels, out_stats, out_centroids = cv2.connectedComponentsWithStats(out_binary,connectivity=4)

    # N includes background CC (0) ...
    # stats are N x 5 = (x, y, w, h, size)
    # centroids are N x 2 = (x, y)

    # 2) Label CCs and get their boundaries on GT image
    gt_n_labels, gt_labels, gt_stats, gt_centroids = cv2.connectedComponentsWithStats(gt_binary, connectivity=4)

    # print((gt_n_labels, out_n_labels))

    # 3) find the size of smallest GT CC
    gt_sizes = []
    for gt_idx in range(gt_n_labels - 1):
        gt_sizes.append(gt_stats[gt_idx, 4])

    min_gt_size = min(gt_sizes)
    # print(gt_sizes)
    # print(min_gt_size)

    # 4) find size of smallest prediction to match with current min IOU_threshold
    min_IOU_threshold = min(IOU_tresholds)
    min_cc_size = min_IOU_threshold * min_gt_size
    # print(min_cc_size)

    # 5) find matches...
    all_pairwise_valid_matches = []
    for out_idx in range(1, out_n_labels):
        if out_stats[out_idx, 4] >= min_cc_size:
            # the CC is large enough to be a valid match .... but is possible that it does not overlap any CC
            # check
            out_cc_x, out_cc_y, out_cc_w, out_cc_h, out_cc_size = out_stats[out_idx]

            out_cc_mask = (out_labels == out_idx)

            # ... against every CC in GT
            for gt_idx in range(1, gt_n_labels):
                gt_cc_x, gt_cc_y, gt_cc_w, gt_cc_h, gt_cc_size = gt_stats[gt_idx]

                # check for overlap between CC (COARSE)
                if ((out_cc_x < gt_cc_x + gt_cc_w and gt_cc_x < out_cc_x + out_cc_w) and
                    (out_cc_y < gt_cc_y + gt_cc_h and gt_cc_y < out_cc_y + out_cc_h)):

                    # quantify pixel-level overlap between CC (FINE)
                    gt_cc_mask = (gt_labels == gt_idx)

                    intersection = np.logical_and(out_cc_mask, gt_cc_mask)
                    union = np.logical_or(out_cc_mask, gt_cc_mask)

                    intersection_size = intersection.sum()
                    union_size = union.sum()

                    # GET IOU
                    IOU = intersection_size / union_size
                    # print((out_idx, gt_idx, IOU))

                    # ONLY consider for matching if IOU is large enough anyway
                    if IOU >= min_IOU_threshold:
                        all_pairwise_valid_matches.append((IOU, gt_idx, out_idx))

             # print((out_idx, cc_with_GT_costs))

            # if has_overlaps:
            # valid_out_CC.append(out_idx + 1)

    # 6) count valid assignments which have IOU over thresholds
    valid_matches_per_threshold = {}
    visualization_images = {}
    for iou_t in IOU_tresholds:
        valid_matches_per_threshold[iou_t] = {"matches": 0}
        if get_visualization:
            visualization_images[iou_t] = np.zeros((gt_binary.shape[0], gt_binary.shape[1], 3), np.uint8)
            visualization_images[iou_t][:, :, 0] = gt_binary.copy()
            visualization_images[iou_t][:, :, 2] = out_binary.copy()
        else:
            visualization_images[iou_t] = None

    # ... sort by decreasing threshold
    all_pairwise_valid_matches = sorted(all_pairwise_valid_matches, reverse=True)
    matched_gt = {}
    matched_out = {}
    for IOU, gt_idx, out_idx in all_pairwise_valid_matches:
        # check if match between two elements not matched before
        if (gt_idx not in matched_gt) and (out_idx not in matched_out):
            # mark both of them as matched
            matched_gt[gt_idx] = True
            matched_out[out_idx] = True

            # only count the match for the cases where IOU surpasses the corresponding threshold
            # if it gets here, it should be at least as large as the first Threshold
            for iou_t in IOU_tresholds:
                if IOU >= iou_t:
                    valid_matches_per_threshold[iou_t]["matches"] += 1

                    if get_visualization:
                        gt_cc_mask = (gt_labels == gt_idx)
                        visualization_images[iou_t][gt_cc_mask, 1] = 255

    """
    
    out_filtered_n = len(valid_out_CC)
    print(out_filtered_n)

    # 4) create cost matrix
    m_size = max(out_filtered_n - 1, gt_n_labels -1 )
    # 3.1) compute all pairwise pixel IOU and use (1 - IOU) as cost
    all_costs = np.ones((m_size, m_size))

    for filtered_idx in range(out_filtered_n - 1):
        out_idx = valid_out_CC[filtered_idx + 1]
        out_cc_x, out_cc_y, out_cc_w, out_cc_h, out_cc_size = out_stats[out_idx]

        out_cc_mask = (out_labels == out_idx)

        for gt_idx in range(gt_n_labels - 1):
            gt_cc_x, gt_cc_y, gt_cc_w, gt_cc_h, gt_cc_size = gt_stats[gt_idx + 1]

            # check for overlap between CC (COARSE)
            if ((out_cc_x < gt_cc_x + gt_cc_w and gt_cc_x < out_cc_x + out_cc_w) and
                (out_cc_y < gt_cc_y + gt_cc_h and gt_cc_y < out_cc_y + out_cc_h)):
                # they overlap at large ... (in range)
                # check for pixel-wise overlap

                gt_cc_mask = (gt_labels == (gt_idx + 1))

                intersection = np.logical_and(out_cc_mask, gt_cc_mask)
                union = np.logical_or(out_cc_mask, gt_cc_mask)

                intersection_size = intersection.sum()
                union_size = union.sum()

                IOU = intersection_size / union_size

                all_costs[filtered_idx, gt_idx] = 1 - IOU

    # print("...munkres...", end="")

    # 5) use hungarian algorithm to find pairwise assignments
    m = Munkres()
    assignments = m.compute(all_costs)

    # 6) count valid assignments which have IOU over threshold hold
    valid_matches_per_threshold = {iou_t: {"matches": 0} for iou_t in IOU_tresholds}
    for filtered_idx, gt_idx in assignments:
        if filtered_idx < out_filtered_n - 1 and gt_idx < gt_n_labels - 1:
            # is a potential valid match ... check for each IOU treshold
            for iou_t in IOU_tresholds:
                if 1 - all_costs[filtered_idx, gt_idx] >= iou_t:
                    valid_matches_per_threshold[iou_t]["matches"] += 1
    """

    # 7) compute precision, recall and F-measure for all IOU thresholds
    for iou_t in IOU_tresholds:
        if gt_n_labels > 1:
            recall = valid_matches_per_threshold[iou_t]["matches"] / (gt_n_labels - 1)
        else:
            recall = 1.0

        if out_n_labels > 1:
            precision = valid_matches_per_threshold[iou_t]["matches"] / (out_n_labels - 1)
        else:
            if gt_n_labels > 1:
                precision = 0.0
            else:
                precision = 1.0

        if recall + precision > 0.0:
            f1 = (2 * recall * precision) / (recall + precision)
        else:
            f1 = 0.0

        valid_matches_per_threshold[iou_t]["recall"] = recall
        valid_matches_per_threshold[iou_t]["precision"] = precision
        valid_matches_per_threshold[iou_t]["f1"] = f1

    # print(valid_matches_per_threshold)
    pixel_matches = np.logical_and(out_binary, gt_binary).sum()
    gt_fg_pixels = gt_binary.sum() / 255
    out_fg_pixels = out_binary.sum() / 255

    pixel_stats = {}
    if gt_fg_pixels > 0:
        pixel_stats["recall"] = pixel_matches / gt_fg_pixels
    else:
        pixel_stats["recall"] = 1.0

    if out_fg_pixels > 0:
        pixel_stats["precision"] = pixel_matches / out_fg_pixels
    else:
        if gt_fg_pixels > 0:
            pixel_stats["precision"] = 0.0
        else:
            pixel_stats["precision"] = 1.0

    if pixel_stats["recall"] + pixel_stats["precision"] > 0.0:
        pixel_stats["f1"] = (2 * pixel_stats["recall"] * pixel_stats["precision"]) / (pixel_stats["recall"] + pixel_stats["precision"])
    else:
        pixel_stats["f1"] = 0.0

    if get_visualization:
        return valid_matches_per_threshold, pixel_stats, visualization_images
    else:
        return valid_matches_per_threshold, pixel_stats


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("\tpython {0:s} config model".format(sys.argv[0]))
        print("Where")
        print("\tconfig\tPath to configuration file")
        print("\tmodel\tPath to network that will be evaluated")
        return

    config = Configuration.from_file(sys.argv[1])
    model_filename = sys.argv[2]

    images_dir = config.get_str("FCN_BINARIZER_PRETRAIN_EVAL_IMAGES_DIR")
    masks_dir = config.get_str("FCN_BINARIZER_PRETRAIN_EVAL_MASKS_DIR")

    bin_threshold = config.get_int("FCN_BINARIZER_PRETRAIN_EVAL_BIN_TRESHOLD", 128)

    all_image_filenames, all_mask_filenames = LectureNet_Util.get_images_w_masks_filenames(images_dir, masks_dir)

    print("... loading model ...")
    lecture_net = FCN_LectureNet.CreateFromConfig(config, 3, False)
    lecture_net.load_state_dict(torch.load(model_filename))
    lecture_net.eval()

    lecture_net = lecture_net.cuda()

    pytorch_total_params = sum(p.numel() for p in lecture_net.parameters() if p.requires_grad)
    print("Total Trainable Parameters in Network: " + str(pytorch_total_params))

    eval_IOU_t = [0.5, 0.75, 0.90]
    count_changed = 0
    with_issues = []
    all_stats = {iou_t: {"recall": [], "precision": [], "f1": []} for iou_t in eval_IOU_t}
    all_pixel_stats = {"recall": [], "precision": [], "f1": []}
    for idx, (img_filename, mask_filename) in enumerate(zip(all_image_filenames[:], all_mask_filenames[:])):
        print("Processing: " + img_filename + " (" + mask_filename + ")", flush=True)

        changed = False

        # load images
        pil_image = Image.open(img_filename)
        o_w, o_h = pil_image.size
        try:
            pil_image = ImageOps.exif_transpose(pil_image)
        except:
            # count this image
            with_issues.append(img_filename)
            count_changed += 1
            # and do not process it further
            continue

        n_w, n_h = pil_image.size

        if pil_image.mode == "CMYK" or pil_image.mode == "L":
            pil_image = pil_image.convert('RGB')
            changed = True

        # print(pil_image.mode)

        if o_w != n_w:
            changed = True

        if changed:
            count_changed += 1
            with_issues.append(img_filename)

        mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)

        # print(mask.shape)

        print("... binarizing ... ", end="")
        binary_image = lecture_net.binarize(pil_image, force_binary=True, binary_treshold=bin_threshold,
                                            apply_sigmoid=True)
        binary_image = 255 - binary_image

        print("... matching ... ", end="")
        image_matches, pixel_stats = compute_matching(binary_image, mask, eval_IOU_t)
        # image_matches, pixel_stats, visuals = compute_matching(binary_image, mask, eval_IOU_t, True)
        print("...done!")

        for IOU_t in eval_IOU_t:
            all_stats[IOU_t]["recall"].append(image_matches[IOU_t]["recall"])
            all_stats[IOU_t]["precision"].append(image_matches[IOU_t]["precision"])
            all_stats[IOU_t]["f1"].append(image_matches[IOU_t]["f1"])

        all_pixel_stats["recall"].append(pixel_stats["recall"])
        all_pixel_stats["precision"].append(pixel_stats["precision"])
        all_pixel_stats["f1"].append(pixel_stats["f1"])

        torch.cuda.empty_cache()

    if len(with_issues) > 0:
        print("\n\nImages with issues fixed: {0:d}".format(count_changed))
        print("List of images with issues")
        for img_name in with_issues:
            print(img_name)

    print("\n\nEvaluation Metrics")
    print("IOU_t\tRec\tPrec\tF1")
    for IOU_t in eval_IOU_t:
        avg_recall = np.mean(all_stats[IOU_t]["recall"])
        avg_precision = np.mean(all_stats[IOU_t]["precision"])
        avg_f1 = np.mean(all_stats[IOU_t]["f1"])

        print("{0:.2f}\t{1:.2f}\t{2:.2f}\t{3:.2f}".format(IOU_t, avg_recall * 100.0, avg_precision * 100.0,
                                                          avg_f1 * 100.0))

    print("\n\nPixel Recall: {0:.2f}".format(np.mean(all_pixel_stats["recall"]) * 100.0))
    print("Pixel Precision: {0:.2f}".format(np.mean(all_pixel_stats["precision"]) * 100.0))
    print("Pixel F1: {0:.2f}".format(np.mean(all_pixel_stats["f1"]) * 100.0))


if __name__ == "__main__":
    main()

