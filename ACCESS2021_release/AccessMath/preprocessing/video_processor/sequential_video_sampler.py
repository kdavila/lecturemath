import cv2

from AM_CommonTools.util.time_helper import TimeHelper


# ===================================================================
# Abstraction for Forced Sequential Video Processing for sampling of
# individual frames. The class relies on external callee to handle
# the special processing of each sampled frame
#
# By: Kenny Davila
#     University at Buffalo
#     2019
#
# ===================================================================

class SequentialVideoSampler:

    def __init__(self, file_list, frame_list):
        self.file_list = file_list
        self.frame_list = sorted(frame_list)
        self.forced_width = None
        self.forced_height = None

        self.time_first = None
        self.time_last = None

    def force_resolution(self, width, height):
        self.forced_width = width
        self.forced_height = height

    def doProcessing(self, video_worker, limit=0, verbose=False):
        # initially....
        width = None
        height = None

        offset_frame = -1
        absolute_frame = 0
        absolute_time = 0.0
        last_frame = None

        next_sample_frame_idx = 0

        if verbose:
            print("Video processing for " + video_worker.getWorkName() + " has begun")

        # for timer...
        timer = TimeHelper()
        timer.startTimer()

        # open video...
        for video_idx, video_file in enumerate(self.file_list):
            try:
                capture = cv2.VideoCapture(video_file)
            except Exception as e:
                # error loading
                raise Exception("The file <" + video_file + "> could not be opened")

            capture_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            capture_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # ...check size ...
            forced_resizing = False
            if width is None:
                # first video....
                # initialize local parameters....
                if self.forced_width is not None:
                    # ...size...
                    width = self.forced_width
                    height = self.forced_height

                    if capture_width != self.forced_width or capture_height != self.forced_height:
                        forced_resizing = True
                else:
                    width = capture_width
                    height = capture_height

                # on the worker class...
                video_worker.initialize(width, height)
            else:
                if self.forced_width is not None:
                    forced_resizing = (capture_width != self.forced_width or capture_height != self.forced_height)
                else:
                    if (width != capture_width) or (height != capture_height):
                        # invalid, all main video files must be the same resolution...
                        raise Exception("All video files on the list must have the same resolution")

            # Read video until the end or until limit has been reached
            while (limit == 0 or offset_frame < limit) and not next_sample_frame_idx >= len(self.frame_list):

                if offset_frame == self.frame_list[next_sample_frame_idx]:
                    # grab and decode next frame since it is in the list
                    flag, frame = capture.read()
                else:
                    # just grab to move forward in the list
                    flag = capture.grab()

                # print("Grab time: " + str(capture.get(cv2.CAP_PROP_POS_FRAMES)))
                # print((valid_grab, flag, type(frame), selection_step, jump_frames))
                if not flag:
                    # end of video reached...
                    break

                if offset_frame == self.frame_list[next_sample_frame_idx]:
                    # continue ..
                    current_time = capture.get(cv2.CAP_PROP_POS_MSEC)
                    current_frame = capture.get(cv2.CAP_PROP_POS_FRAMES)

                    if forced_resizing:
                        frame = cv2.resize(frame, (self.forced_width, self.forced_height))

                    frame_time = absolute_time + current_time
                    frame_idx = int(absolute_frame + current_frame)
                    video_worker.handleFrame(frame, last_frame, video_idx, frame_time, current_time, frame_idx)

                    if verbose :
                        print("Frames Processed = {0:d}, Video Time = {1:s}".format(offset_frame, TimeHelper.stampToStr(frame_time)))

                    last_frame = frame

                    next_sample_frame_idx += 1
                    if next_sample_frame_idx >= len(self.frame_list):
                        # end of sample reached ...
                        break

                offset_frame += 1

            # at the end of the processing of current video
            capture.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
            video_length = capture.get(cv2.CAP_PROP_POS_MSEC)
            video_frames = capture.get(cv2.CAP_PROP_POS_FRAMES)

            absolute_time += video_length
            absolute_frame += video_frames

        # processing finished...
        video_worker.finalize()

        # end time counter...
        timer.endTimer()

        if verbose:
            print("Video processing for " + video_worker.getWorkName() + " completed: " +
                  TimeHelper.stampToStr(timer.lastElapsedTime() * 1000.0))


