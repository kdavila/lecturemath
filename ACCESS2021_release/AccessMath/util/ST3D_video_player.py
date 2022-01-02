
import time

import numpy as np


# TODO: Move to it's own file ???
class ST3D_SequentialReader:
    ReadBinary = 0
    ReadReconstructed = 1
    ReadStable = 2

    def __init__(self, cc_stability, st3D, read_mode):
        self.cc_stability = cc_stability
        self.st3D = st3D
        self.offset = 0
        self.read_mode = read_mode
        self.last_frame_idx = 0
        self.last_frame_time = 0.0

    def __compute_binary(self):
        frame_ccs = self.cc_stability.cc_idx_per_frame[self.offset]
        binary = self.cc_stability.rebuilt_binary_frame(frame_ccs)

        return binary

    def ___read_binary(self):
        binary = self.__compute_binary()

        frame = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
        frame[:, :, 0] = binary
        frame[:, :, 1] = binary
        frame[:, :, 2] = binary

        frame = 255 - frame

        return frame

    def __compute_reconstructed(self):
        current_frame_idx = self.st3D.frame_indices[self.offset]
        visible_groups = self.st3D.groups_in_frame_range(current_frame_idx, current_frame_idx)
        frame_ccs = self.st3D.get_CC_instances(visible_groups, current_frame_idx)
        reconstructed = self.cc_stability.rebuilt_binary_frame(enumerate(frame_ccs))

        return reconstructed

    def ___read_reconstructed(self):
        frame = np.zeros((self.st3D.height, self.st3D.width, 3), dtype=np.uint8)

        reconstructed_binary = self.__compute_reconstructed()

        frame[:, :, 0] = reconstructed_binary
        frame[:, :, 1] = reconstructed_binary
        frame[:, :, 2] = reconstructed_binary

        frame = 255 - frame

        return frame

    def __read_stable(self):
        # get raw binary
        raw_binary = self.__compute_binary()

        # get reconstructed ...
        reconstructed_binary = self.__compute_reconstructed()

        frame = np.zeros((self.st3D.height, self.st3D.width, 3), dtype=np.uint8)

        frame[:, :, 0] = reconstructed_binary
        frame[:, :, 1] = reconstructed_binary
        frame[:, :, 2] = raw_binary

        return frame

    def read(self):
        if self.offset < len(self.st3D.frame_indices):
            if self.read_mode == ST3D_SequentialReader.ReadBinary:
                frame = self.___read_binary()
            elif self.read_mode == ST3D_SequentialReader.ReadReconstructed:
                frame = self.___read_reconstructed()
            elif self.read_mode == ST3D_SequentialReader.ReadStable:
                frame = self.__read_stable()
            else:
                raise Exception("Unknown Sequential Reader Mode")

            # copy current properties ...
            self.last_frame_idx = self.st3D.frame_indices[self.offset]
            self.last_frame_time = self.st3D.frame_times[self.offset]

            # move ....
            self.offset += 1

            return True, frame
        else:
            # ended, cannot continue reading ....
            return False, None

    def get_current_time(self):
        return self.last_frame_time

    def get_current_frame_index(self):
        return self.last_frame_idx

    def set_current_frame_index(self, new_frame_idx):
        if new_frame_idx <= self.st3D.frame_indices[0]:
            # re-start from the very beginning
            self.offset = 0
            self.last_frame_idx = 0
            self.last_frame_time = 0.0
        else:
            if new_frame_idx >= self.st3D.frame_indices[-1]:
                # reach the end  ...
                self.offset = len(self.st3D.frame_indices) - 1
            else:
                # do a binary search for the right offset ...
                start_offset = 0
                end_offset = len(self.st3D.frame_indices) - 1

                while start_offset < end_offset:
                    mid_offset = int((start_offset + end_offset) / 2)
                    if new_frame_idx < self.st3D.frame_indices[mid_offset]:
                        # continue search on the left ...
                        end_offset = mid_offset - 1
                    else:
                        if start_offset == mid_offset:
                            # probably found ...
                            break
                        else:
                            # continue search on the right ....
                            start_offset = mid_offset

                self.offset = start_offset

            self.last_frame_idx = self.st3D.frame_indices[self.offset]
            self.last_frame_time = self.st3D.frame_times[self.offset]

from .base_video_player import BaseVideoPlayer

class ST3D_VideoPlayer(BaseVideoPlayer):
    FrameCache = 500  # around 3 GB (500, un-compressed)

    def __init__(self, cc_stability, st3d):
        BaseVideoPlayer.__init__(self)

        # the original cc stability estimator
        self.cc_stability = cc_stability
        # the spatio-temporal 3D structure to visualize
        self.st3D = st3d

        self.reader = ST3D_SequentialReader(cc_stability, st3d, ST3D_SequentialReader.ReadReconstructed)

        self.width = self.st3D.width
        self.height = self.st3D.height

        self.current_time = None
        self.last_time = time.time()

        self.total_frames = self.st3D.frame_indices[-1]
        self.total_length = self.st3D.frame_times[-1]

        self.cache_images = []
        self.cache_times = []
        self.cache_frames = []
        self.cache_offset = 0.0
        self.cache_pos = 0

        self.black_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)


    def get_frame(self):
        self.current_time = time.time()
        # in seconds ...
        delta = (self.current_time - self.last_time) * self.play_speed
        self.last_time = self.current_time

        if self.playing:
            # update last frame ...
            # use milliseconds (just like opencv does)
            self.play_abs_position += delta * 1000.0

            # while desired frame not in cache ....
            while not self.__frame_in_cache(self.play_abs_position) and not self.end_reached:
                # ... update cache
                self.__extract_next_frame()

            # get frame
            self.last_frame_img, self.last_frame_idx = self.__get_cached_frame(self.play_abs_position)

            if self.frame_changed_callback is not None:
                if len(self.cache_frames) > 0:
                    self.frame_changed_callback(int(self.cache_frames[self.cache_pos]), self.cache_times[self.cache_pos])

        frame = self.apply_frame_zoom()

        return frame, self.last_frame_idx

    def __frame_in_cache(self, abs_time):
        if len(self.cache_times) == 0:
            return False

        return self.cache_offset <= abs_time <= self.cache_times[-1]

    def __get_cached_frame(self, abs_time):
        if self.end_reached:
            if len(self.cache_images) > 0:
                return self.cache_images[-1], self.cache_frames[-1]
            else:
                return self.black_frame, self.total_frames

        # assume forward movement only
        while self.cache_times[self.cache_pos] < abs_time:
            self.cache_pos += 1

        return self.cache_images[self.cache_pos], self.cache_frames[self.cache_pos]

    def __extract_next_frame(self):
        # get the next frame ...
        pre_time = time.time()
        flag, next_frame = self.reader.read()
        post_time = time.time()
        # print(post_time - pre_time)

        # frame can be re-sized here if needed ....

        if not flag:
            # failed to get frame ....
            self.end_reached = True
            next_frame = None

        if next_frame is not None:
            # update cache ...

            last_time = self.reader.get_current_time()
            frame_number = self.reader.get_current_frame_index()

            self.cache_times.append(last_time)
            self.cache_images.append(next_frame)
            self.cache_frames.append(frame_number)

            while len(self.cache_times) > ST3D_VideoPlayer.FrameCache:
                self.cache_offset = self.cache_times[0]
                if self.cache_pos > 0:
                    self.cache_pos -= 1
                del self.cache_times[0]
                del self.cache_images[0]
                del self.cache_frames[0]
        else:
            frame_number = self.total_frames

        return next_frame, frame_number

    def set_position_frame(self, new_abs_frame, notify_listeners):
        # check that frame is within video boundaries ... or force it to be otherwise ...
        if new_abs_frame >= self.total_frames:
            new_abs_frame = self.total_frames - 2

        if new_abs_frame < 0:
            new_abs_frame = 0

        # first, check if the frame is still in the cache ...
        if len(self.cache_frames) > 0 and (self.cache_frames[0] <= new_abs_frame <= self.cache_frames[-1]):
            # already on cache, just move the time to the right frame

            start_cache_pos = 0
            end_cache_pos = len(self.cache_frames)

            while start_cache_pos < end_cache_pos:
                mid_cache_pos = int((start_cache_pos + end_cache_pos) / 2)

                if new_abs_frame < self.cache_frames[mid_cache_pos]:
                    # move to the left ...
                    end_cache_pos = mid_cache_pos - 1
                else:
                    # check ....
                    if self.cache_frames[mid_cache_pos] == new_abs_frame or mid_cache_pos == start_cache_pos:
                        # found exactly or approximately
                        break
                    else:
                        # move to the right ...
                        start_cache_pos = mid_cache_pos

            offset = start_cache_pos

            self.cache_pos = offset
            self.play_abs_position = self.cache_times[self.cache_pos]
            self.last_frame_img = self.cache_images[self.cache_pos]
            self.last_frame_idx = self.cache_frames[self.cache_pos]
        else:
            # reset cache
            self.cache_frames = []
            self.cache_images = []
            self.cache_times = []
            self.cache_pos = 0

            # find desired position ....
            self.reader.set_current_frame_index(new_abs_frame)

            # update video location and cache start
            self.play_abs_position = self.reader.get_current_time()
            self.cache_offset = self.play_abs_position

            #print("here")

            # read next frame
            self.last_frame_img, self.last_frame_idx = self.__extract_next_frame()

            #print("done")

        if self.frame_changed_callback is not None and notify_listeners:
            self.frame_changed_callback(int(new_abs_frame), self.play_abs_position)

        self.last_time = time.time()

    def set_binary_mode(self):
        # change mode ...
        self.reader.read_mode = ST3D_SequentialReader.ReadBinary
        # reset cache ...
        self.cache_frames = []
        # re-read next frame ...
        self.set_position_frame(self.reader.last_frame_idx, True)

    def set_reconstructed_mode(self):
        # change mode ...
        self.reader.read_mode = ST3D_SequentialReader.ReadReconstructed
        # reset cache ...
        self.cache_frames = []
        # re-read next frame ...
        self.set_position_frame(self.reader.last_frame_idx, True)

    def set_stable_cc_mode(self):
        # change mode ...
        self.reader.read_mode = ST3D_SequentialReader.ReadStable
        # reset cache ...
        self.cache_frames = []
        # re-read next frame ...
        self.set_position_frame(self.reader.last_frame_idx, True)

