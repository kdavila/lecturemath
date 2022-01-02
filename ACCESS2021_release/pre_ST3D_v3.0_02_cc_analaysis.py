


#============================================================================
# Preprocessing Model for ST3D indexing - V 3.0
#
# Kenny Davila
# - Created:  March 18, 2017
# - Modified: December 31, 2021
#============================================================================

import sys

# from AccessMath.preprocessing.config.parameters import Parameters
from AccessMath.preprocessing.content.helper import Helper
from AccessMath.preprocessing.content.cc_stability_estimator import CCStabilityEstimator
from AccessMath.preprocessing.user_interface.console_ui_process import ConsoleUIProcess

def process_input(process, input_data):
    frame_times, frame_indices, compressed_frames = input_data

    # De-compress images
    print("Decompressing input...")
    binary_frames = Helper.decompress_binary_images(compressed_frames)

    height, width = binary_frames[0].shape

    #Parameters
    #cc_stability = Parameters....
    cc_min_recall = process.configuration.get_float("CC_STABILITY_MIN_RECALL", 0.925)
    cc_min_precision = process.configuration.get_float("CC_STABILITY_MIN_PRECISION", 0.925)
    cc_max_gap = process.configuration.get_int("CC_STABILITY_MAX_GAP", 85)
    estimator = CCStabilityEstimator(width, height, cc_min_recall, cc_min_precision, cc_max_gap, True)

    print("Processing frames...")
    # process
    for frame_idx, frame in enumerate(binary_frames):
        estimator.add_frame(frame, True)

    # finish
    estimator.finish_processing()

    return frame_times, frame_indices, estimator

def main():
    # usage check
    if not ConsoleUIProcess.usage_with_config_check(sys.argv):
        return

    process = ConsoleUIProcess.FromConfigPath(sys.argv[1], sys.argv[2:], "BINARIZATION_OUTPUT",
                                              "CC_STABILITY_OUTPUT")
    if not process.initialize():
        return

    process.start_input_processing(process_input)

    print("Finished!")

if __name__ == "__main__":
    main()
