
import sys

import numpy as np
import matplotlib.pyplot as plt

from AM_CommonTools.configuration.configuration import Configuration
from AccessMath.util.misc_helper import MiscHelper
from AccessMath.data.meta_data_DB import MetaDataDB
from AccessMath.annotation.lecture_annotation import LectureAnnotation

from AccessMath.preprocessing.content.helper import Helper
from AccessMath.preprocessing.content.video_segmenter import VideoSegmenter

def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("\tpython {0:s} config training".format(sys.argv[0]))
        print("Where")
        print("\tconfig:\tPath to Configuration File")
        print("\ttraining:\t 1 for training set, 0 for testing set")
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

    output_dir = config.get_str("OUTPUT_PATH")
    temporal_segments_dir = output_dir + "/" + database.output_temporal
    images_dir = output_dir + "/" + database.output_images

    binary_prefix = config.get_str("BINARIZATION_OUTPUT")

    if use_training:
        dataset_name = "training"
    else:
        dataset_name = "testing"

    current_dataset = database.datasets[dataset_name]

    for current_lecture in current_dataset:
        print("Processing: " + current_lecture.title)

        # read binary data ...
        input_filename = temporal_segments_dir + "/" + binary_prefix + current_lecture.title + ".dat"
        binary_data = MiscHelper.dump_load(input_filename)
        frame_times, frame_indices, compressed_frames = binary_data

        print("...Decompressing input...")
        all_binary = Helper.decompress_binary_images(compressed_frames)

        print("...Computing sums...")
        all_sums = VideoSegmenter.compute_binary_sums(all_binary)

        # read ground truth for this lecture ...
        input_filename = (output_dir + "/" + database.output_annotations + "/" +
                          database.name + "_" + current_lecture.title.lower() + ".xml")
        annotation = LectureAnnotation.Load(input_filename, False)

        gt_segments = [0] + annotation.video_segments + [annotation.total_frames]

        # make the plot ...
        y = np.array(all_sums)

        # Plot the results
        fig = plt.figure(figsize=(8, 6), dpi=300)

        ax1 = fig.add_subplot(111)

        max_y_value = y.max() * 1.10

        X = np.arange(len(all_sums))
        ax1.fill_between(X, y, facecolor="#7777DD", alpha=0.5)

        plt.title("Ideal Segments: " + current_lecture.title)

        for segment_frame_idx in range(len(gt_segments)):
            split_x = int((gt_segments[segment_frame_idx] / gt_segments[-1]) * len(all_sums))
            plt.plot(np.array([split_x, split_x]), np.array([0, max_y_value]), c="r", linewidth=1)

        plt.xlabel("data")
        plt.ylabel("binary sums")

        #plt.legend()

        out_filename = images_dir + "/GT_intervals_" + current_lecture.title + ".png"
        plt.savefig(out_filename, dpi=300)

        plt.close()

if __name__ == "__main__":
    main()


