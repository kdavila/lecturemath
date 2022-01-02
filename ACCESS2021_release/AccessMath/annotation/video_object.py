
from scipy import interpolate

from .video_object_location import VideoObjectLocation


class VideoObject:
    XMLNamespace = ''

    # This can be expended to more arbitrary shapes in the future now that the location is polygon
    ShapeAlignedRectangle = 0
    ShapeQuadrilateral = 1
    ShapePolygon = 2

    def __init__(self, id, name, shape_type):
        self.id = id
        self.name = name
        self.locations = []
        self.shape_type = shape_type

    def make_polygon_split_copy(self):
        new_poly_object = VideoObject(self.id, self.name, VideoObject.ShapePolygon)
        for loc in self.locations:
            split_polygon = loc.get_split_polygon()

            new_loc = VideoObjectLocation(loc.visible, loc.frame, loc.abs_time, split_polygon, label=loc.label)
            new_poly_object.locations.append(new_loc)

        return new_poly_object

    def first_frame(self):
        return self.locations[0].frame

    def last_frame(self):
        return self.locations[-1].frame

    def is_rectangle(self):
        return self.shape_type == VideoObject.ShapeAlignedRectangle

    def polygon_points(self):
        return self.locations[0].n_points()

    def update_timeline(self, frame_scale_factor, time_scale_factor):
        for loc in self.locations:
            loc.frame = int(round(loc.frame * frame_scale_factor))
            loc.abs_time *= time_scale_factor

    def find_location_idx(self, frame):
        loc_min = 0
        loc_max = len(self.locations) - 1
        while loc_min <= loc_max:
            loc_mid = int((loc_min + loc_max) / 2.0)

            if self.locations[loc_mid].frame == frame:
                return loc_mid
            elif self.locations[loc_mid].frame < frame:
                loc_min = loc_mid + 1
            else:
                if loc_max == loc_mid:
                    break
                else:
                    loc_max = loc_mid

        return loc_min

    def set_location_at(self, frame, abs_time, visible, polygon_points):
        loc_idx = self.find_location_idx(frame)

        if (loc_idx >= len(self.locations)) or (self.locations[loc_idx].frame != frame):

            if loc_idx >= 1:
                prev_label = self.locations[loc_idx - 1].label
            else:
                prev_label = None

            # does not exist, create
            location = VideoObjectLocation(visible, frame, abs_time, polygon_points, prev_label)
            # insert at desired idx ...
            self.locations.insert(loc_idx, location)

            # Key-frame was added
            return True
        else:
            # udpate existing ...
            self.locations[loc_idx].update(visible, polygon_points)

            # an existing Key-frame was updated
            return False

    def del_location_at(self, frame):
        loc_idx = self.find_location_idx(frame)

        if (loc_idx >= len(self.locations)) or (self.locations[loc_idx].frame != frame):
            # does not exist
            return False
        else:
            # exists
            del self.locations[loc_idx]
            return True

    def get_location_at(self, frame, out_range, interpolate=True):
        # out of range -> None if not extrapolate else
        if len(self.locations) == 0:
            raise Exception("Cannot estimate out of range, no existing locations")

        loc_idx = self.find_location_idx(frame)

        if (loc_idx >= len(self.locations)) or (loc_idx == 0 and self.locations[loc_idx].frame != frame):
            if not out_range:
                # out of range
                return None
            else:
                if loc_idx == 0:
                    # use the first
                    return self.locations[0]
                else:
                    # use the last
                    return self.locations[-1]
        else:
            # check if exact ...
            if self.locations[loc_idx].frame == frame:
                # exact match, no interpolation required ...
                return self.locations[loc_idx]
            else:
                if interpolate:
                    # interpolate ...
                    return VideoObjectLocation.interpolate(self.locations[loc_idx - 1], self.locations[loc_idx], frame)
                else:
                    # do not interpolate, return last frame ..
                    return self.locations[loc_idx - 1]

    def all_unique_labels(self):
        all_labels = {}
        for loc in self.locations:
            if loc.label in all_labels:
                all_labels[loc.label] += 1
            else:
                all_labels[loc.label] = 1

        return all_labels

    def toXML(self):
        result = "  <VideoObject>\n"
        result += "    <Id>" + self.id + "</Id>\n"
        result += "    <Name>" + self.name + "</Name>\n"
        result += "    <Shape>" + str(self.shape_type) + "</Shape>\n"
        result += "    <VideoObjectLocations>\n"
        for location in self.locations:
            result += location.toXML("        ")
        result += "    </VideoObjectLocations>\n"

        result += "  </VideoObject>\n"

        return result

    def get_export_info(self, total_frames):
        # known information ...
        all_frame_idxs = []
        all_times = []
        for loc in self.locations:
            all_frame_idxs.append(loc.frame)
            all_times.append(loc.abs_time)

        # create inter/extra-polator function
        time_f = interpolate.interp1d(all_frame_idxs, all_times, fill_value='extrapolate')

        all_frames = []
        for frame_idx in range(total_frames + 1):
            frame_time = time_f([frame_idx])[0]

            loc = self.get_location_at(frame_idx, True, True)
            known = self.locations[0].frame <= frame_idx <= self.locations[-1].frame

            known_str = "1" if known else "0"
            visible_str = "1" if loc.visible else "0"

            out_label = "" if loc.label is None else loc.label

            general_info = frame_idx, frame_time, known_str, visible_str, out_label
            polygon_info = loc.polygon_points.ravel()

            all_frames.append((general_info, polygon_info))

        return all_frames

    def export_CSV(self, filename, total_frames):
        # get all the frame-wise data to export
        all_frames_info = self.get_export_info(total_frames)

        # prepare output headers ....
        all_lines = []
        header = ["frame_idx", "frame_time", "known", "visible", "label"]
        for p_idx in range(self.locations[0].polygon_points.shape[0]):
            header += ["p_" + str(p_idx) + "_x", "p_" + str(p_idx) + "_y"]

        all_lines.append(",".join(header) + "\n")

        # for each frame to export ... create a string representation
        for frame_idx in range(total_frames + 1):
            general_info, polygon_info = all_frames_info[frame_idx]

            frame_idx, frame_time, known_str, visible_str, out_label = general_info

            line_info = [str(frame_idx), str(frame_time), known_str, visible_str, out_label]
            line_info += [str(val) for val in polygon_info]

            all_lines.append(",".join(line_info) + "\n")

        with open(filename, "w", encoding="utf-8") as output_file:
            output_file.writelines(all_lines)


    @staticmethod
    def fromXML(root):
        # general properties
        object_id = root.find(VideoObject.XMLNamespace + 'Id').text
        object_name = root.find(VideoObject.XMLNamespace + 'Name').text

        shape_root = root.find(VideoObject.XMLNamespace + 'Shape')
        if shape_root is None:
            print("Warning: Legacy Video Object Annotation found")
            shape_type = VideoObject.ShapeAlignedRectangle
        else:
            shape_type = int(shape_root.text)

        if not shape_type in [VideoObject.ShapeAlignedRectangle, VideoObject.ShapeQuadrilateral,
                              VideoObject.ShapePolygon]:
            raise Exception("VideoObject: Invalid Shape Type found!")

        video_object = VideoObject(object_id, object_name, shape_type)

        # locations
        locations_root = root.find(VideoObject.XMLNamespace + 'VideoObjectLocations')
        locations_xml = locations_root.findall(VideoObject.XMLNamespace + 'VideoObjectLocation')
        
        for location_xml in locations_xml:
            video_object.locations.append(VideoObjectLocation.fromXML(location_xml))

        return video_object


