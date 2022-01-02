import time
from collections import OrderedDict
import cv2
import numpy as np

from AccessMath.preprocessing.video_processor.image_list_processor import ImageListGenerator

from .base_video_player import BaseVideoPlayer

class ImageListCache(OrderedDict):
    def __init__(self, max_size=500):
        self.max_size = max_size
        super(ImageListCache, self).__init__()

    def __setitem__(self, key, value, **kwargs):
        if len(self) == self.max_size:
            self.popitem(last=False)
        super(ImageListCache, self).__setitem__(key, value)
        self.move_to_end(key)


class ImageListVideoPlayer(ImageListGenerator, BaseVideoPlayer):
    FrameCache = 500
    def __init__(self, folder, forced_resolution=None, file_extension='.jpg'):
        # super(ImageListVideoPlayer, self).__init__(folder, extension=file_extension, preload=False)
        ImageListGenerator.__init__(self, folder, extension=file_extension, preload=False)
        BaseVideoPlayer.__init__(self)

        if forced_resolution is None:
            self.forced_width, self.forced_height = None, None
        else:
            self.forced_width, self.forced_height = forced_resolution

        self.current_time = None
        self.last_time = time.time()

        self.sorted_keys = sorted(self.metadata.keys(), key=lambda x: int(x))
        self.virtual_len = self.metadata[self.sorted_keys[-1]]['frame_idx']
        self.actual_len = len(self.metadata) - 1
        self.virtual_frame_to_image_map = {self.metadata[key]['frame_idx'] : k for k, key in enumerate(self.sorted_keys)}
        self.virtual_time_to_image_map = {self.metadata[key]['abs_time'] : k for k, key in enumerate(self.sorted_keys)}
        self.virtual_time_to_virtual_frame = {self.metadata[key]['abs_time'] : self.metadata[key]['frame_idx'] for key in self.sorted_keys}
        self.frame_idxs = np.asarray(list(self.virtual_frame_to_image_map.keys())[1:])
        self.frame_idxs.sort()
        self.frame_times = np.asarray(list(self.virtual_time_to_image_map.keys())[1:])
        self.frame_times.sort()

        self.cache = ImageListCache(self.FrameCache)

        self.height, self.width = self.get_image_dims()

        self.total_frames = self.virtual_len
        self.total_length = self.metadata[self.sorted_keys[-1]]['abs_time']
        # print(self.sorted_keys)
        # print(self.virtual_frame_to_image_map)
        # print(self.frame_idxs)

    @staticmethod
    def interpolate_time(t1, t2, f1, f2, f):
        t = (t2 - t1) * (f - f1) / (f2 - f1)
        t += t1
        return t

    @staticmethod
    def interpolate_frame(f1, f2, t1, t2, t):
        f = (f2 - f1) * (t - t1) / (t2 - t1)
        f += f1
        return int(f)
        
    def get_image_dims(self):
        im = self.__getitem__(0)
        h, w, c = im.shape
        return h, w

    def find_nearest_virtual_frame(self, virtual_frame):
        return self.frame_idxs[np.argmin(np.abs(self.frame_idxs - virtual_frame))]

    def find_nearest_virtual_time(self, virtual_time):
        return self.frame_times[np.argmin(np.abs(self.frame_times - virtual_time))]

    def find_virtual_time_by_frame(self, virtual_frame):
        nearest_virtual_idx = self.find_nearest_virtual_frame(virtual_frame)
        nearest_virtual_time = self.metadata[str(nearest_virtual_idx)]['abs_time']
        if virtual_frame < nearest_virtual_idx and virtual_frame >= 0:
            neighbour_virtual_idx = self.frame_idxs[np.argmin(np.abs(self.frame_idxs - virtual_frame)) - 1]
            neighbour_virtual_time = self.metadata[str(neighbour_virtual_idx)]['abs_time']
            virtual_time = self.interpolate_time(neighbour_virtual_time, nearest_virtual_time,
                                                 neighbour_virtual_idx, nearest_virtual_idx, virtual_frame)
        elif virtual_frame > nearest_virtual_idx and virtual_frame <= self.total_frames:
            neighbour_virtual_idx = self.frame_idxs[np.argmin(np.abs(self.frame_idxs - virtual_frame)) + 1]
            neighbour_virtual_time = self.metadata[str(neighbour_virtual_idx)]['abs_time']
            virtual_time = self.interpolate_time(nearest_virtual_time, neighbour_virtual_time,
                                                 nearest_virtual_idx, neighbour_virtual_idx, virtual_frame)
        else:
            virtual_time = nearest_virtual_time
        return virtual_time

    def find_virtual_frame_by_time(self, virtual_time):
        nearest_virtual_time = self.find_nearest_virtual_time(virtual_time)
        nearest_virtual_frame = self.virtual_time_to_virtual_frame[nearest_virtual_time]
        if virtual_time < nearest_virtual_time and virtual_time > 0.:
            neighbour_virtual_time =  self.frame_times[np.argmin(np.abs(self.frame_times - virtual_time)) - 1]
            neighbour_virtual_frame =  self.virtual_time_to_virtual_frame[neighbour_virtual_time]
            virtual_frame = self.interpolate_frame(neighbour_virtual_frame, nearest_virtual_frame,
                                                 neighbour_virtual_time, nearest_virtual_time, virtual_time)
        elif virtual_time > nearest_virtual_time and virtual_time < self.total_length:
            neighbour_virtual_time = self.frame_times[np.argmin(np.abs(self.frame_times - virtual_time)) + 1]
            neighbour_virtual_frame = self.virtual_time_to_virtual_frame[neighbour_virtual_time]
            virtual_frame = self.interpolate_frame(nearest_virtual_frame, neighbour_virtual_frame,
                                                 nearest_virtual_time,neighbour_virtual_time, virtual_time)
        else:
            virtual_frame = nearest_virtual_frame
        return virtual_frame

    def __len__(self):
        return self.virtual_len

    def __getitem__(self, item):
        nearest_virtual_frame = self.find_nearest_virtual_frame(item)
        image_idx = self.virtual_frame_to_image_map[nearest_virtual_frame] - 1
        im = super(ImageListVideoPlayer, self).__getitem__(image_idx)
        if self.forced_width is not None:
            im = cv2.resize(im, (self.forced_width, self.forced_height), interpolation=cv2.INTER_AREA)
        return im

    def set_position_frame(self, frame, notify_listeners):
        virtual_time = self.find_virtual_time_by_frame(frame)
        nearest_virtual_time = self.find_nearest_virtual_time(virtual_time)

        # check cache
        if nearest_virtual_time in self.cache:
            self.last_frame_img = self.cache[nearest_virtual_time]
            self.last_frame_idx = frame
        else:
            im = self[frame]
            self.cache[nearest_virtual_time] = im
            self.last_frame_img = im
            self.last_frame_idx = frame

        self.play_abs_position = virtual_time

        if self.frame_changed_callback is not None and notify_listeners:
            self.frame_changed_callback(int(frame), self.play_abs_position)

        self.last_time = time.time()

    def get_frame(self):
        self.current_time = time.time()
        # in seconds ...
        delta = (self.current_time - self.last_time) * self.play_speed
        self.last_time = self.current_time
        # print(1.0 / delta)

        if self.playing:
            # update last frame ...
            # use milliseconds (just like opencv does)
            self.play_abs_position += delta * 1000.0
            nearest_virtual_time = self.find_nearest_virtual_time(self.play_abs_position)
            # get frame from cache if present if not update cache
            if nearest_virtual_time in self.cache:
                self.last_frame_img = self.cache[nearest_virtual_time]
            else:
                image_idx = self.virtual_time_to_image_map[nearest_virtual_time] - 1
                im = super(ImageListVideoPlayer, self).__getitem__(image_idx)
                self.cache[nearest_virtual_time] = im
                self.last_frame_img = im
            self.last_frame_idx = self.find_virtual_frame_by_time(self.play_abs_position)

            if self.frame_changed_callback is not None:
                self.frame_changed_callback(self.last_frame_idx, self.play_abs_position)

        # last_frame_img must be updated here ....
        frame = self.apply_frame_zoom()

        return frame, self.last_frame_idx

if __name__ == '__main__':
    from AccessMath.preprocessing.config.parameters import Parameters
    imvp = ImageListVideoPlayer(Parameters.Output_FrameExport + 'lecture_01/JPEGImages', file_extension=Parameters.Output_FrameExport_ImgExtension)
    imvp.playing = True
    # for v_idx in range(1, len(imvp)):
    #     nearest_virtual_idx = imvp.find_nearest_virtual_frame(v_idx)
    #     nearest_image_idx = imvp.virtual_frame_to_image_map[imvp.find_nearest_virtual_frame(v_idx)] - 1
    #     nearest_image_path = imvp.ims[imvp.virtual_frame_to_image_map[imvp.find_nearest_virtual_frame(v_idx)] - 1]
    #     virtual_time = imvp.find_virtual_time_by_frame(v_idx)
    #     print(v_idx, nearest_virtual_idx, nearest_image_idx, nearest_image_path, virtual_time)
    #     cv2.imshow('current frame', imvp[v_idx])
    #     # cv2.waitKey(33)
    # cv2.destroyAllWindows()
    frame, frame_idx = imvp.get_frame()
    print(frame_idx, imvp.find_nearest_virtual_frame(frame_idx))
    time.sleep(120)
    frame, frame_idx = imvp.get_frame()
    print(frame_idx, imvp.find_nearest_virtual_frame(frame_idx))

