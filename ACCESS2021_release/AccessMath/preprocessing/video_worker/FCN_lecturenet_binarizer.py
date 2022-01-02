
import cv2

from PIL import Image

class FCN_LectureNet_Binarizer:
    def __init__(self, lecture_net):
        self.width = 0
        self.height = 0

        self.frame_count = 0

        # binarizer
        self.lecture_net = lecture_net

        self.last_binary = None
        self.last_text = None
        self.last_rec = None

        self.frame_times = None
        self.frame_indices = None
        self.compressed_frames = None

        self.debug_mode = False
        self.debug_start = 0.0
        self.debug_end = 0.0
        self.debug_out_dir = None
        self.debug_video_name = ""

    def initialize(self, width, height):
        self.width = width
        self.height = height

        self.frame_count = 0

        self.frame_times = []
        self.frame_indices = []
        self.compressed_frames = []

    def set_debug_mode(self, active, start_time, end_time, out_dir, video_name):
        self.debug_mode = active
        self.debug_start = start_time
        self.debug_end = end_time
        self.debug_out_dir = out_dir
        self.debug_video_name = video_name

    def handleFrame(self, frame, last_frame, v_index, abs_time, rel_time, abs_frame_idx):
        self.frame_count += 1

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        binary, text_mask, rec_img = self.lecture_net.binarize(pil_image, return_others=True, force_binary=True)

        # invert!
        binary = 255 - binary

        flag, raw_data = cv2.imencode(".png", binary)

        self.last_binary = binary
        self.last_text = text_mask
        self.last_rec = rec_img

        self.compressed_frames.append(raw_data)
        self.frame_indices.append(abs_frame_idx)
        self.frame_times.append(abs_time)

        if self.debug_mode:
            if self.debug_start <= abs_time <= self.debug_end:
                self.debug_frame(binary)

    def debug_frame(self, binary):
        out_name = self.debug_out_dir + "/binary_" + self.debug_video_name + "_" + str(self.frame_count) + ".png"
        cv2.imwrite(out_name, binary)


    def getWorkName(self):
        return "FCN_LectureNet Frame Binarizer"

    def finalize(self):
        pass

