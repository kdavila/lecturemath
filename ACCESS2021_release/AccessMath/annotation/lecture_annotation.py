
from xml.etree import ElementTree as ET

import numpy as np
from shapely.geometry.point import Point

from AccessMath.annotation.video_object import VideoObject
from AccessMath.annotation.video_object_location import VideoObjectLocation
from AccessMath.annotation.drawing_info import DrawingInfo

class LectureAnnotation:
    XMLNamespace = ''

    def __init__(self, db_name, lecture_title, output_file, video_files, total_frames, drawing_info):
        self.db_name = db_name
        self.title = lecture_title
        self.output_file = output_file
        self.video_files = video_files
        self.total_frames = total_frames
        self.drawing_info = drawing_info

        self.frame_width = None
        self.frame_height = None

        self.video_objects = {}
        self.video_segments = []
        self.video_segment_keyframes = []

    def set_frame_resolution(self, width, height):
        self.frame_width = width
        self.frame_height = height

    def contains(self, name):
        return name in self.video_objects

    def find_temporal_closest_point_container(self, point, frame_idx):
        if not isinstance(point, Point):
            point = Point(point)

        intersections = []
        for name in self.video_objects:
            video_object = self.video_objects[name]

            # get estimated video object location for this frame
            object_loc = video_object.get_location_at(frame_idx, True)

            # check if object contains point ...
            if object_loc.polygon.intersects(point):
                # check temporal distance ...
                if video_object.first_frame() <= frame_idx <= video_object.last_frame():
                    # object is on this frame
                    temporal_distance = 0
                else:
                    if frame_idx < video_object.first_frame():
                        # object appears later in the video  at this location
                        temporal_distance = video_object.first_frame() - frame_idx
                    else:
                        # object appears earlier in the video at this location
                        temporal_distance = frame_idx - video_object.last_frame()

                intersections.append((temporal_distance, name))

        if len(intersections) == 0:
            # not objects found at this location
            return None
        else:
            # return the closest in time (forward or backwards)
            intersections = sorted(intersections)
            tempo_dist, object_name = intersections[0]

            return self.video_objects[object_name]

    def get_next_object_name_correlative(self, name_prefix):
        max_correlative = None
        for object_name in self.video_objects:
            if object_name[:len(name_prefix)] == name_prefix:
                str_corr = object_name[len(name_prefix):]
                try:
                    int_corr = int(str_corr)
                except:
                    # remainder does not represent a valid numeric string, ignore
                    continue

                if max_correlative is None:
                    max_correlative = int_corr
                elif int_corr > max_correlative:
                    max_correlative = int_corr

        if max_correlative is None:
            # first object with this prefix
            return 1
        else:
            # use max correlative + 1
            return max_correlative + 1

    def __getitem__(self, item):
        return self.video_objects[item]

    def get(self, object_name):
        return self.video_objects[object_name]

    def add_object(self, id, name, shape_type, frame, abs_time, polygon_points):
        if name in self.video_objects:
            return False

        self.video_objects[id] = VideoObject(id, name, shape_type)
        self.video_objects[id].set_location_at(frame, abs_time, True, polygon_points)

        return True

    def rename_object(self, old_id, new_id, new_name):
        if new_id in self.video_objects:
            return False

        # ... reference to object ....
        # .... copy ref ...
        self.video_objects[new_id] = self.video_objects[old_id]
        # .... remove old ref ...
        del self.video_objects[old_id]
        # .... change object name
        self.video_objects[new_id].id = new_id
        self.video_objects[new_id].name = new_name

        return True

    def remove_object(self, object_name):
        if object_name not in self.video_objects:
            return False

        # ... remove from video objects
        del self.video_objects[object_name]

        return True

    def project_object_location(self, loc):
        assert isinstance(loc, VideoObjectLocation)

        assert isinstance(self.drawing_info, DrawingInfo)

        off_x = self.drawing_info.player_render_bbox[0] - self.drawing_info.canvas_bbox[0]
        off_y = self.drawing_info.player_render_bbox[1] - self.drawing_info.canvas_bbox[1]

        # note that these values should be the same (if aspect ratio is kept)
        scale_x = self.frame_width / self.drawing_info.player_render_bbox[2]
        scale_y = self.frame_height / self.drawing_info.player_render_bbox[3]

        proj_points = (loc.polygon_points - np.array([[off_x, off_y]]))
        proj_points[:, 0] *= scale_x
        proj_points[:, 1] *= scale_y

        return VideoObjectLocation(loc.visible, loc.frame, loc.abs_time, proj_points)

    def generate_metadata_header_xml(self):
        xml_string = "  <Database>" + self.db_name + "</Database>\n"
        xml_string += "  <Lecture>" + self.title + "</Lecture>\n"
        xml_string += "  <Filename>" + self.output_file + "</Filename>\n"
        xml_string += "  <VideoFiles>\n"
        for filename in self.video_files:
            xml_string += "     <VideoFile>" + filename + "</VideoFile>\n"
        xml_string += "  </VideoFiles>\n"

        return xml_string

    def generate_video_segments_xml(self):
        tempo_segments = [0] + self.video_segments + [self.total_frames]
        xml_string = "  <VideoSegments>\n"
        for idx in range(len(self.video_segments) + 1):
            xml_string += "    <VideoSegment>\n"
            xml_string += "        <Start>" + str(tempo_segments[idx]) + "</Start>\n"
            xml_string += "        <End>" + str(tempo_segments[idx + 1]) + "</End>\n"
            xml_string += "    </VideoSegment>\n"
        xml_string += "  </VideoSegments>\n"

        return xml_string

    def generate_keyframes_xml(self, include_objects, keyframe_times=None):
        xml_string = "  <VideoKeyFrames>\n"
        for idx, frame_idx in enumerate(self.video_segment_keyframes):
            xml_string += "    <VideoKeyFrame>\n"
            xml_string += "       <Index>" + str(frame_idx) + "</Index>\n"

            if keyframe_times is not None:
                xml_string += "       <AbsTime>" + str(keyframe_times[idx]) + "</AbsTime>\n"

            if include_objects:
                xml_string += "       <VideoObjects>\n"

                # for each object ....
                for object_name in self.video_objects:
                    # get location of object at current key-frame
                    loc = self.video_objects[object_name].get_location_at(frame_idx, False)
                    shape = self.video_objects[object_name].shape_type

                    # only add if object is visible at current key-frame
                    if loc is not None and loc.visible:
                        proj_loc = self.project_object_location(loc)

                        object_xml = "          <VideoObject>\n"
                        object_xml += "              <Name>" + object_name + "</Name>\n"
                        object_xml += "              <Shape>" + str(shape) + "</Shape>\n"
                        object_xml += "              <Polygon>\n"
                        for x, y in proj_loc.polygon_points:
                            object_xml += "                 <Point>\n"
                            object_xml += "                    <X>" + str(x) + "</X>\n"
                            object_xml += "                    <Y>" + str(y) + "</Y>\n"
                            object_xml += "                 </Point>\n"

                        object_xml += "              </Polygon>\n"
                        object_xml += "          </VideoObject>\n"

                        xml_string += object_xml

                xml_string += "       </VideoObjects>\n"

            xml_string += "    </VideoKeyFrame>\n"

        xml_string += "  </VideoKeyFrames>\n"

        return xml_string

    def generate_data_xml(self):
        xml_string = "<Annotations>\n"

        # general meta-data
        xml_string += self.generate_metadata_header_xml()

        # add ViewPort coordinates info ...
        xml_string += self.drawing_info.generate_xml()

        xml_string += "  <VideoObjects>\n"
        for name in sorted(list(self.video_objects.keys())):
            xml_string += self.video_objects[name].toXML()
        xml_string += "  </VideoObjects>\n"

        xml_string += self.generate_video_segments_xml()

        # save key-frames without object info (full object info already saved)
        xml_string += self.generate_keyframes_xml(False)

        xml_string += "</Annotations>\n"

        return xml_string

    def generate_export_xml(self, keyframe_times):
        xml_string = "<Annotations>\n"

        # general meta-data
        xml_string += self.generate_metadata_header_xml()

        # segments
        xml_string += self.generate_video_segments_xml()

        # key frames with object data ...
        xml_string += self.generate_keyframes_xml(True, keyframe_times)

        xml_string += "</Annotations>\n"

        return xml_string

    def update_timeline(self, new_frame_count, old_frame_count=None, new_time=None, old_time=None):
        if old_frame_count is None:
            # by default, use the value in the annotation itself
            frame_scale_factor = new_frame_count / self.total_frames
        else:
            # use the provided value to compute the new scale
            frame_scale_factor = new_frame_count / old_frame_count

        if new_time is not None and old_time is not None:
            time_scale_factor = new_time / old_time
        else:
            # no temporal scaling
            time_scale_factor = 1.0

        for object_name in self.video_objects:
            self.video_objects[object_name].update_timeline(frame_scale_factor, time_scale_factor)

        new_video_segments = [int(round(val * frame_scale_factor)) for val in self.video_segments]
        self.video_segments = new_video_segments

        new_video_segment_keyframes = [int(round(val * frame_scale_factor)) for val in self.video_segment_keyframes]
        self.video_segment_keyframes = new_video_segment_keyframes

        # replace length
        self.total_frames = new_frame_count

        return frame_scale_factor, time_scale_factor

    def save(self, output_path=None):
        xml_data = self.generate_data_xml()

        if output_path is None:
            output_path = self.output_file

        out_file = open(output_path, "w")
        out_file.write(xml_data)
        out_file.close()

    @staticmethod
    def Show_XML_Metadata(database_name, lecture_title, output_file, video_parts):
        print("- Database: " + str(database_name))
        print("- Lecture: " + str(lecture_title))
        print("- Output: " + str(output_file))
        print("- Videos: ")
        for file_video in video_parts:
            print("\t" + file_video)

    @staticmethod
    def Load(filename, verbose=True):
        tree = ET.parse(filename)
        root = tree.getroot()

        namespace = LectureAnnotation.XMLNamespace

        database_name = root.find(LectureAnnotation.XMLNamespace + 'Database').text
        lecture_title = root.find(LectureAnnotation.XMLNamespace + 'Lecture').text
        output_file = root.find(LectureAnnotation.XMLNamespace + 'Filename').text

        video_files = []
        file_videos = root.find(LectureAnnotation.XMLNamespace + 'VideoFiles')
        for file_video in file_videos.findall(LectureAnnotation.XMLNamespace + 'VideoFile'):
            video_files.append(file_video.text)

        if verbose:
            # Show meta-data just for validation purposes
            print("Loading data:")
            LectureAnnotation.Show_XML_Metadata(database_name, lecture_title, output_file, video_files)

        # load video segments ...
        xml_video_segments_root = root.find(namespace + "VideoSegments")
        xml_video_segment_objects = xml_video_segments_root.findall(namespace + "VideoSegment")
        tempo_split_points = []
        tempo_ends_points = []
        for xml_video_segment_object in xml_video_segment_objects:
            split_point = int(xml_video_segment_object.find(VideoObject.XMLNamespace + 'Start').text)
            end_point = int(xml_video_segment_object.find(VideoObject.XMLNamespace + 'End').text)
            tempo_split_points.append(split_point)
            tempo_ends_points.append(end_point)

        total_frames = max(tempo_ends_points)
        tempo_split_points = sorted(tempo_split_points)
        if 0 in tempo_split_points:
            tempo_split_points.remove(0)

        drawing_info = DrawingInfo.from_XML(root, namespace)

        annotation = LectureAnnotation(database_name, lecture_title, output_file, video_files, total_frames,
                                       drawing_info)

        annotation.video_segments = tempo_split_points

        # load video objects
        xml_video_objects_root = root.find(namespace + 'VideoObjects')
        xml_video_objects = xml_video_objects_root.findall(namespace + 'VideoObject')
        msg_object = " -> Loading object: {0:s} ({1:d} Key-frames)"
        for xml_video_object in xml_video_objects:
            # load logical object ...
            video_object = VideoObject.fromXML(xml_video_object)

            if verbose:
                print(msg_object.format(video_object.name, len(video_object.locations)))

            # logical object
            annotation.video_objects[video_object.id] = video_object

        if verbose:
            print(" -> A total of {0:d} video objects where loaded!".format(len(annotation.video_objects)))

        # load key-frames ...
        xml_video_keyframes_root = root.find(namespace + "VideoKeyFrames")
        xml_video_keyframes_objects = xml_video_keyframes_root.findall(namespace + "VideoKeyFrame")
        tempo_keyframes = []
        for xml_video_keyframe_object in xml_video_keyframes_objects:
            frame_idx = int(xml_video_keyframe_object.find(namespace + "Index").text)
            tempo_keyframes.append(frame_idx)

        tempo_keyframes = sorted(tempo_keyframes)
        annotation.video_segment_keyframes = tempo_keyframes

        return annotation
