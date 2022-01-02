
import time


class BaseVideoPlayer:
    MaxZoomFactor = 2

    def __init__(self):
        self.width = None
        self.height = None

        self.total_frames = None
        self.total_length = None

        self.play_speed = 1.0
        self.play_abs_position = 0.0
        self.playing = False
        self.end_reached = False

        self.last_frame_img = None
        self.last_frame_idx = None

        self.frame_changed_callback = None

        # allowing zoom
        self.zoom_factor = 0
        self.panning_x_factor = 0
        self.panning_y_factor = 0

    def play(self):
        self.current_time = None
        self.last_time = time.time()

        self.playing = True
        self.end_reached = False

    def pause(self):
        self.playing = False

    def apply_frame_zoom(self):
        if self.zoom_factor == 0:
            frame = self.last_frame_img
        else:
            cut_width = self.visible_width()
            cut_height = self.visible_height()

            cut_x = self.visible_left()
            cut_y = self.visible_top()

            frame = self.last_frame_img[cut_y:cut_y + cut_height, cut_x:cut_x + cut_width]

        return frame

    def visible_width(self):
        return int(self.width / pow(2, self.zoom_factor))

    def visible_height(self):
        return int(self.height / pow(2, self.zoom_factor))

    def visible_left(self):
        return int((self.width - self.visible_width()) * self.panning_x_factor)

    def visible_top(self):
        return int((self.height - self.visible_height()) * self.panning_y_factor)

    def zoom_increase(self):
        if self.zoom_factor < BaseVideoPlayer.MaxZoomFactor:
            self.zoom_factor += 1
            return True
        else:
            return False

    def zoom_decrease(self):
        if self.zoom_factor > 0:
            self.zoom_factor -= 1
            return True
        else:
            return False

    def set_horizontal_panning(self, new_x_panning):
        if 0.0 <= new_x_panning <= 1.0:
            self.panning_x_factor = new_x_panning

    def set_vertical_panning(self, new_y_panning):
        if 0.0 <= new_y_panning <= 1.0:
            self.panning_y_factor = new_y_panning

