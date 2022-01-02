
#============================================================================
# Preprocessing Model for ST3D indexing - V 3.0
#
# Kenny Davila
# - Created:  March 10, 2021
# - Modified: December 30, 2021
#
#============================================================================

import sys

import torch

from AccessMath.preprocessing.user_interface.console_ui_process import ConsoleUIProcess

from AccessMath.lecturenet_v1.FCN_lecturenet import FCN_LectureNet
from AccessMath.preprocessing.video_worker.FCN_lecturenet_binarizer import FCN_LectureNet_Binarizer

def get_worker(process):

    # use FCN LectureNet
    print("... loading model ...")

    output_dir = process.configuration.get_str("OUTPUT_PATH")
    model_dir = output_dir + "/" + process.configuration.get_str("BINARIZATION_FCN_LECTURENET_DIR")
    model_filename = model_dir + "/" + process.configuration.get_str("BINARIZATION_FCN_LECTURENET_FILENAME")

    use_cuda = process.configuration.get("FCN_BINARIZER_USE_CUDA", True)

    lecture_net = FCN_LectureNet.CreateFromConfig(process.configuration, 3, False)
    lecture_net.load_state_dict(torch.load(model_filename))
    lecture_net.eval()

    # use cuda ...
    if use_cuda:
        lecture_net = lecture_net.cuda()

    worker_binarizer = FCN_LectureNet_Binarizer(lecture_net)

    debug_mode = process.configuration.get("BINARIZATION_DEBUG_MODE", False)
    debug_end_time = process.configuration.get_int("BINARIZATION_DEBUG_END_TIME", 50000)

    worker_binarizer.set_debug_mode(debug_mode, 0, debug_end_time, process.img_dir, process.current_lecture.title)

    return worker_binarizer


def get_results(worker):
    # using Lecture net ... delete ...
    del worker.lecture_net
    # now, empty CUDA cache ...
    torch.cuda.empty_cache()

    return (worker.frame_times, worker.frame_indices, worker.compressed_frames)


def main():
    # usage check
    if not ConsoleUIProcess.usage_with_config_check(sys.argv):
        return

    process = ConsoleUIProcess.FromConfigPath(sys.argv[1], sys.argv[2:], None, "BINARIZATION_OUTPUT")
    if not process.initialize():
        return

    fps = process.configuration.get_float("SAMPLING_FPS", 1.0)
    process.start_video_processing(fps, get_worker, get_results, 0, True, True)

    print("finished")


if __name__ == "__main__":
    main()

