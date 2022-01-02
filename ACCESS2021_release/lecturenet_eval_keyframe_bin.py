
import os
import sys
import time

import cv2
import numpy as np

from PIL import Image

import torch

from lecturenet_train_02_train_binarizer import FCN_LectureNet

from AM_CommonTools.configuration.configuration import Configuration
from AM_CommonTools.util.time_helper import TimeHelper

from AccessMath.annotation.keyframe_annotation import KeyFrameAnnotation
from AccessMath.data.meta_data_DB import MetaDataDB
from AccessMath.evaluation.evaluator import Evaluator
from AccessMath.evaluation.eval_parameters import EvalParameters


def main():
    if len(sys.argv) < 4:
        print("Usage:")
        print("\tpython {0:s} config model dataset".format(sys.argv[0]))
        print("Where:")
        print("\tconfig:\tAccessMath Configuration File used to train network")
        print("\tmodel:\tPath to trained model to evaluate")
        print("\tdataset:\t")
        print("\t\t0 - Training Dataset")
        print("\t\t1 - Testing Dataset")
        return

    config_filename = sys.argv[1]
    model_filename = sys.argv[2]
    try:
        dataset_int = int(sys.argv[3])
        if dataset_int == 0:
            dataset = "training"
        elif dataset_int == 1:
            dataset = "testing"
        else:
            print("Invalid value for parameter: dataset")
            return
    except:
        print("Invalid value for parameter: dataset")
        return

    # read the configuration file ....
    config = Configuration.from_file(config_filename)

    # load the database
    try:
        database = MetaDataDB.from_file(config.get_str("VIDEO_DATABASE_PATH"))
    except:
        print("Invalid database file")
        return

    output_dir = config.get_str("OUTPUT_PATH")
    binary_save_dir = config.get_str("FCN_BINARIZER_SAVE_BINARY_PATH", ".")
    use_cuda = config.get("FCN_BINARIZER_USE_CUDA", True)

    start_loading = time.time()

    print("... loading model ...")
    lecture_net = FCN_LectureNet.CreateFromConfig(config, 3, False)
    lecture_net.load_state_dict(torch.load(model_filename))
    lecture_net.eval()

    if use_cuda:
        lecture_net = lecture_net.cuda()

    print("... loading data ...")
    all_keyframes, binarized_keyframes = KeyFrameAnnotation.LoadDatasetKeyframes(output_dir, database, dataset)
    fake_unique_groups, fake_cc_group, fake_segments = KeyFrameAnnotation.GenerateFakeKeyframeInfo(all_keyframes)

    pytorch_total_params = sum(p.numel() for p in lecture_net.parameters() if p.requires_grad)
    print("Total Trainable Parameters in Network: " + str(pytorch_total_params))

    end_loading = time.time()

    start_binarizing = time.time()

    last_lecture = None
    lecture_offset = -1
    current_dataset = database.get_dataset(dataset)

    for idx, bin_kf in enumerate(binarized_keyframes):
        if bin_kf.lecture != last_lecture:
            last_lecture = bin_kf.lecture
            lecture_offset += 1

        print("binarizing kf #" + str(idx) + ", from " + current_dataset[lecture_offset].title, end="\r", flush=True)

        pil_image = Image.fromarray(cv2.cvtColor(bin_kf.raw_image, cv2.COLOR_RGB2BGR))

        binary_image = lecture_net.binarize(pil_image, force_binary=True)

        # filtered_bin = 255 - binary_image
        filtered_bin = binary_image

        bin_kf.binary_image = np.zeros((filtered_bin.shape[0], filtered_bin.shape[1], 3), dtype=np.uint8)
        bin_kf.binary_image[:, :, 0] = filtered_bin
        bin_kf.binary_image[:, :, 1] = filtered_bin
        bin_kf.binary_image[:, :, 2] = filtered_bin

        bin_kf.update_binary_cc(False)

        if config.get("FCN_BINARIZER_SAVE_BINARY", True):
            binary_dir = binary_save_dir + "/FCN"
            binary_dir += "/" + current_dataset[lecture_offset].title + "/binary"

            os.makedirs(binary_dir, exist_ok=True)
            out_name = binary_dir + "/" + str(bin_kf.idx) + ".png"

            cv2.imwrite(out_name, bin_kf.binary_image)

        if use_cuda:
            torch.cuda.empty_cache()

    end_binarizing = time.time()

    # run evaluation metrics ...
    print("Computing final evaluation metrics....")

    # Summary level metrics ....
    start_evaluation = time.time()

    EvalParameters.UniqueCC_global_tran_window = 1
    EvalParameters.UniqueCC_min_precision = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 0.95]
    EvalParameters.UniqueCC_min_recall = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 0.95]
    EvalParameters.Report_Summary_Show_Counts = False
    EvalParameters.Report_Summary_Show_AVG_per_frame = False
    EvalParameters.Report_Summary_Show_Globals = True

    all_scope_metrics, scopes = Evaluator.compute_summary_metrics(fake_segments, all_keyframes, fake_unique_groups,
                                                            fake_cc_group, fake_segments, binarized_keyframes, False)

    for scope in scopes:
        print("")
        print("Metrics for scope: " + scope)
        print("      \t      \tRecall\t      \t       \tPrecision")
        print("Min R.\tMin P.\tE + P\tE. Only\tP. Only\tE + P\tE. Only\tP. Only\tBG. %\tNo BG P.")
        scope_metrics = all_scope_metrics[scope]

        recall_percent_row = "{0:.2f}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.2f}"
        prec_percent_row = "{0:.2f}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.2f}"

        for all_metrics in scope_metrics:
            metrics = all_metrics["recall_metrics"]

            recall_str = recall_percent_row.format(all_metrics["min_cc_recall"] * 100.0,
                                                   all_metrics["min_cc_precision"] * 100.0,
                                                   metrics["recall"] * 100.0, metrics["only_exact_recall"] * 100.0,
                                                   metrics["only_partial_recall"] * 100.0)

            metrics = all_metrics["precision_metrics"]

            prec_str = prec_percent_row.format(metrics["precision"] * 100.0, metrics["only_exact_precision"] * 100.0,
                                               metrics["only_partial_precision"] * 100.0,
                                               metrics["global_bg_unmatched"] * 100.0,
                                               metrics["no_bg_precision"] * 100.0)

            print(recall_str + "\t" + prec_str)

    # pixel level metrics
    pixel_metrics = Evaluator.compute_pixel_binary_metrics(all_keyframes, binarized_keyframes)
    print("Pixel level metrics")
    for key in sorted(pixel_metrics.keys()):
        print("{0:s}\t{1:.2f}".format(key, pixel_metrics[key] *100.0))

    end_evaluation = time.time()

    end_everything = time.time()

    print("Total loading time: " + TimeHelper.secondsToStr(end_loading - start_loading))
    print("Total binarization time: " + TimeHelper.secondsToStr(end_binarizing - start_binarizing))
    print("Total evaluation time: " + TimeHelper.secondsToStr(end_evaluation - start_evaluation))
    print("Total Time: " + TimeHelper.secondsToStr(end_everything - start_loading))


if __name__ == "__main__":
    main()
