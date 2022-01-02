import numpy as np

class KeyFramePortion:
    def __init__(self, x, y, w, h, binary_image=None, dark=False):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.binary = binary_image
        self.dark = dark

    def get_XML_string(self, include_binary=False, indent=""):
        xml_string = indent + "<KeyFramePortion>\n"
        xml_string += indent + "    <X>" + str(self.x) + "</X>\n"
        xml_string += indent + "    <Y>" + str(self.y) + "</Y>\n"
        xml_string += indent + "    <W>" + str(self.w) + "</W>\n"
        xml_string += indent + "    <H>" + str(self.h) + "</H>\n"
        xml_string += indent + "    <Dark>" + ("1" if self.dark else "0") + "</Dark>\n"
        if include_binary:
            xml_string += indent + "    <Binary>" + str((self.binary / 255).astype(np.uint8).tolist()) + "</Binary>\n"
        xml_string += indent + "</KeyFramePortion>\n"

        return xml_string

    def get_area(self):
        return self.w * self.h

    def overlaps(self, r_x, r_y, r_w, r_h):
        return (self.x < r_x + r_w and r_x < self.x + self.w) and (self.y < r_y + r_h and r_y < self.y + self.h)

    def is_contained(self, r_x, r_y, r_w, r_h):
        return (r_x <= self.x and self.x + self.w <= r_x + r_w) and (r_y <= self.y and self.y + self.h <= r_h + r_h)

    def black_pixel_count(self):
        total_white = self.binary.sum() / 255
        return self.binary.shape[0] * self.binary.shape[1] - total_white

    def invert_region(self):
        self.binary = 255 - self.binary

    def clear_region(self, r_x, r_y, r_w, r_h):
        start_x = max(r_x, self.x) - self.x
        start_y = max(r_y, self.y) - self.y
        end_x = min(r_x + r_w, self.x + self.w) - self.x
        end_y = min(r_y + r_h, self.y + self.h) - self.y

        self.binary[start_y:end_y, start_x:end_x] = 255

    @staticmethod
    def Copy(other):
        assert isinstance(other, KeyFramePortion)

        return KeyFramePortion(other.x, other.y, other.w, other.h, other.binary.copy(), other.dark)