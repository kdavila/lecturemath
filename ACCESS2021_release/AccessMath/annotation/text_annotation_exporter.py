
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np

# from .video_object import VideoObject
# from .video_object_location import VideoObjectLocation
from AccessMath.annotation.lecture_annotation import LectureAnnotation

class TextAnnotationExporter:
    ExportModeAllPerFrame = 0
    ExportModeUniqueBoxes = 1
    ExportModeFullSynthetic = 2

    def __init__(self, annotation, object_prefixes, speaker_name, max_speaker_intersection, export_mode, export_dir,
                 export_images=False):
        self.export_mode = export_mode
        self.img_width = None
        self.img_height = None

        """
        # Source render info ...
        self.canvas_loc = canvas_loc
        self.render_loc = render_loc
        self.render_size = render_size

        self.proj_off_x = None
        self.proj_off_y = None
        self.proj_scale_x = None
        self.proj_scale_y = None
        """

        # directory where results will be stored ...
        self.export_dir = export_dir
        self.export_img_dir = export_dir + "/JPEGImages"
        self.export_xml_dir = export_dir + "/Annotations"
        self.export_bin_dir = export_dir + "/Binary"
        self.export_images= export_images # flag to control if images need to be exported
        self.export_img_format = "png"

        self.annotation = annotation
        self.object_prefixes = object_prefixes
        self.speaker_name = speaker_name
        self.max_speaker_inter = max_speaker_intersection

        self.text_objects = []
        self.speaker = None

        # for unique-objects export mode
        self.exported_text_objects = None
        self.unique_objects_xml_tree = None

        # filter text annotations
        if self.annotation.video_objects is not None:
            for video_object_name in self.annotation.video_objects:
                video_object = self.annotation.video_objects[video_object_name]

                if TextAnnotationExporter.CheckTextObject(video_object, self.object_prefixes):
                    # a text region object found ...
                    self.text_objects.append(video_object)
                else:
                    if video_object.id.lower() == self.speaker_name.lower():
                        # speaker object found ...
                        self.speaker = video_object

    def initialize(self, width, height, prepare_dirs=True):
        self.img_width = width
        self.img_height = height

        self.annotation.set_frame_resolution(width, height)

        # TODO: Will anything be required for synthetic data?

        if self.export_mode == TextAnnotationExporter.ExportModeUniqueBoxes:
            self.exported_text_objects = {}
            self.unique_objects_xml_tree = ET.Element('annotation')

        # prepare export root ...
        if prepare_dirs:
            os.makedirs(self.export_img_dir, exist_ok=True)
            os.makedirs(self.export_xml_dir, exist_ok=True)

            if self.export_mode == TextAnnotationExporter.ExportModeFullSynthetic:
                os.makedirs(self.export_bin_dir, exist_ok=True)

    def getWorkName(self):
        return "Text Annotation Exporter"

    def frame_visible_bboxes_state(self, frame_idx):
        # find speaker location
        if self.speaker is None:
            speaker_loc = None
        else:
            speaker_loc = self.speaker.get_location_at(frame_idx, False)

        # for each text object ...
        not_occluded_bboxes = []
        occluded_bboxes = []
        for text_object in self.text_objects:
            # get interpolated location at current frame (if present)
            text_loc = text_object.get_location_at(frame_idx, False)
            text_name = text_object.name

            # check if text box is present on current frame ..
            if text_loc is not None and text_loc.visible:
                # text is in the current frame ...

                # check if occluded by the speaker
                if (speaker_loc is None) or (not speaker_loc.visible):
                    # no speaker on the image
                    # iou = 0.0
                    int_area_prc = 0.0
                else:
                    # speaker is on the image, check for occlusion as defined by IOU
                    # iou = speaker_loc.IOU(text_loc)
                    int_area_prc = text_loc.intersection_percentage(speaker_loc)

                # project ...
                proj_loc = self.annotation.project_object_location(text_loc)
                # proj_loc = self.project_object_location(text_loc)

                # mark as either occluded or not ...
                # if iou < 0.0001:
                if int_area_prc <= self.max_speaker_inter:
                    # not enough intersection with speaker ...
                    not_occluded_bboxes.append((text_name, proj_loc.get_polygon_points()))
                else:
                    # intersects with the speaker too much, deemed as occluded!
                    occluded_bboxes.append((text_name, proj_loc.get_polygon_points()))

        return speaker_loc, not_occluded_bboxes, occluded_bboxes

    def debug_show_polygons(self, frame, frame_idx, speaker_loc, not_occluded_polygons, occluded_polygons):
        frame_copy = frame.copy()

        not_occluded = []
        for text_name, polygon in not_occluded_polygons:
            points = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            not_occluded.append(points)

        cv2.polylines(frame_copy, not_occluded, True, (0, 255, 0), thickness=2)

        occluded = []
        for text_name, polygon in occluded_polygons:
            points = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            occluded.append(points)

        cv2.polylines(frame_copy, occluded, True, (0, 0, 255), thickness=2)

        if speaker_loc is not None and speaker_loc.visible:
            proj_speaker = self.annotation.project_object_location(speaker_loc)

            points = proj_speaker.polygon_points.copy().astype(np.int32).reshape((-1, 1, 2))

            cv2.polylines(frame_copy, [points], True, (255, 0, 0), thickness=2)

        debug_filename = "{0:s}/debug/{1:d}.png".format(self.export_dir, frame_idx)
        # print(debug_filename)
        cv2.imwrite(debug_filename, frame_copy)

    def export_all_by_frame(self, frame, frame_idx, not_occluded_polygons, binary=None):
        # Output file names ...
        out_img_filename = "{0:s}/{1:d}.{2:s}".format(self.export_img_dir, frame_idx, self.export_img_format)
        out_bin_filename = "{0:s}/{1:d}.{2:s}".format(self.export_bin_dir, frame_idx, self.export_img_format)
        out_xml_filename = "{0:s}/{1:d}.xml".format(self.export_xml_dir, frame_idx)

        # Export Bounding Boxes in XML format ...
        # ... get XML for non-occluded boxes
        # ... Save boxes ...
        xml_tree = TextAnnotationExporter.generate_XML_objects(out_img_filename, self.img_width, self.img_height,
                                                               not_occluded_polygons)
        xml_tree.write(out_xml_filename)

        # ... save image(s)  ... including binary if given ...
        if self.export_images:
            if self.export_img_format.lower() == 'png':
                cv2.imwrite(out_img_filename, frame)
                if binary is not None:
                    cv2.imwrite(out_bin_filename, binary)
            else:
                cv2.imwrite(out_img_filename, frame, (cv2.IMWRITE_JPEG_QUALITY, 100))
                if binary is not None:
                    cv2.imwrite(out_bin_filename, binary, (cv2.IMWRITE_JPEG_QUALITY, 100))

    def export_unique_objects(self, frame, frame_idx, not_occluded_polygons):
        # check which objects are initially visible and can be exported (not exported before)
        for text_name, polygon in not_occluded_polygons:
            array_poly = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))

            x1 = max(0, int(array_poly[:, 0, 0].min()))
            y1 = max(0, int(array_poly[:, 0, 1].min()))
            x2 = min(int(self.img_width), int(array_poly[:, 0, 0].max()))
            y2 = min(int(self.img_height), int(array_poly[:, 0, 1].max()))

            tempo_mask = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
            cv2.fillPoly(tempo_mask, [array_poly], (255, 255, 255))

            region_bbox = (x1, y1, x2, y2)
            _, region_img = cv2.imencode(".png", frame[y1:y2, x1:x2])
            _, region_mask = cv2.imencode(".png", tempo_mask[y1:y2, x1:x2, 0])
            current_object = (frame_idx, region_bbox, array_poly, region_img, region_mask)

            if text_name not in self.exported_text_objects:
                # mark as exported ...
                self.exported_text_objects[text_name] = [current_object]
            else:
                self.exported_text_objects[text_name].append(current_object)

    def handleFrame(self, frame, last_frame, video_idx, frame_time, current_time, frame_idx):
        # Compute and export sample frame metadata
        speaker_loc, not_occluded_polygons, occluded_polygons = self.frame_visible_bboxes_state(frame_idx)

        # total_in_frame = len(occluded_boxes) + len(not_occluded_bboxes)
        # print("-> Text count: {0:d} / {1:d}".format(len(not_occluded), total_in_frame))

        # self.debug_show_polygons(frame, frame_idx, speaker_loc, not_occluded_polygons, occluded_polygons)

        if self.export_mode == TextAnnotationExporter.ExportModeAllPerFrame:
            # export the frame (single image) and a file with all the metadata for all GT bboxes
            self.export_all_by_frame(frame, frame_idx, not_occluded_polygons)
        elif self.export_mode == TextAnnotationExporter.ExportModeUniqueBoxes:
            # only export unique bounding boxes that are not occluded (first time seen)
            self.export_unique_objects(frame, frame_idx, not_occluded_polygons)
        else:
            raise Exception("Invalid export mode")

    def append_XML_unique_object(self, filepath, object_name, polygon):
        object_xml = ET.SubElement(self.unique_objects_xml_tree, 'object')

        # ... image location ...
        folder_name, image_filename = os.path.split(filepath)
        filename = ET.SubElement(object_xml, 'filename')
        filename.text = image_filename
        folder = ET.SubElement(object_xml, 'folder')
        folder.text = folder_name

        name = ET.SubElement(object_xml, 'name')
        name.text = object_name

        polygon_xml = ET.SubElement(object_xml, 'polygon')
        for p_idx, (px, py) in enumerate(polygon):
            px_xml = ET.SubElement(polygon_xml, 'x' + str(p_idx))
            px_xml.text = str(px)
            py_xml = ET.SubElement(polygon_xml, 'y' + str(p_idx))
            py_xml.text = str(py)

    def finalize_unique_text_boxes(self):
        # compute a single "best image" for each unique object
        for text_name in self.exported_text_objects:
            object_instances = self.exported_text_objects[text_name]

            tempo_decoded_images = []
            tempo_decoded_masks = []

            # first, obtain object global boundaries
            all_x1, all_y1, all_x2, all_y2 = [], [], [], []
            for frame_idx, region_bbox, array_poly, region_img, region_mask in object_instances:
                x1, y1, x2, y2 = region_bbox
                all_x1.append(x1)
                all_y1.append(y1)
                all_x2.append(x2)
                all_y2.append(y2)

                tempo_decoded_images.append(cv2.imdecode(region_img, cv2.IMREAD_COLOR))
                tempo_decoded_masks.append(cv2.imdecode(region_mask, cv2.IMREAD_GRAYSCALE))

            gb_x1 = min(all_x1)
            gb_y1 = min(all_y1)
            gb_x2 = max(all_x2)
            gb_y2 = max(all_y2)

            # compute average image ...
            avg_img = np.zeros((gb_y2 - gb_y1, gb_x2 - gb_x1, 3), dtype=np.float64)
            avg_count = np.zeros((gb_y2 - gb_y1, gb_x2 - gb_x1), dtype=np.int32)
            for idx, (frame_idx, (x1, y1, x2, y2), array_poly, region_img, region_mask) in enumerate(object_instances):
                off_x = x1 - gb_x1
                off_y = y1 - gb_y1
                end_y = off_y + tempo_decoded_images[idx].shape[0]
                end_x = off_x + tempo_decoded_images[idx].shape[1]

                avg_img[off_y:end_y, off_x:end_x] += tempo_decoded_images[idx]
                avg_count[off_y:end_y, off_x:end_x] += (tempo_decoded_masks[idx] / 255).astype(np.int32)

            avg_mask = avg_count > 0
            avg_img[avg_mask, 0] /= avg_count[avg_mask]
            avg_img[avg_mask, 1] /= avg_count[avg_mask]
            avg_img[avg_mask, 2] /= avg_count[avg_mask]

            avg_img = avg_img.astype(np.uint8)

            # find the image with the smallest difference to the average image
            all_mse = []
            for idx, (frame_idx, (x1, y1, x2, y2), array_poly, region_img, region_mask) in enumerate(object_instances):
                off_x = x1 - gb_x1
                off_y = y1 - gb_y1
                end_y = off_y + tempo_decoded_images[idx].shape[0]
                end_x = off_x + tempo_decoded_images[idx].shape[1]

                diff = avg_img[off_y:end_y, off_x:end_x].astype(np.int32) - tempo_decoded_images[idx].astype(np.int32)
                mse = np.power(diff, 2).mean()

                all_mse.append((mse, idx))

            all_mse = sorted(all_mse)

            # use the smallest difference frame ...
            final_idx = all_mse[0][1]
            _, final_bbox, final_poly, _, _ = object_instances[final_idx]
            final_image = tempo_decoded_images[final_idx]

            # TODO: accept other image formats as output using corresponding class member for img format
            out_img_filename = "{0:s}/{1:s}.png".format(self.export_img_dir, text_name)
            # ... add to xml ...
            self.append_XML_unique_object(out_img_filename, text_name, final_poly[:, 0, :])
            # ... save image ...
            cv2.imwrite(out_img_filename, final_image)

        # save XML results
        out_xml_filename = "{0:s}/text_objects.xml".format(self.export_xml_dir)

        annotation = ET.ElementTree(self.unique_objects_xml_tree)
        annotation.write(out_xml_filename)

    def finalize(self):
        if self.export_mode == TextAnnotationExporter.ExportModeUniqueBoxes:
            self.finalize_unique_text_boxes()

    @staticmethod
    def CheckTextObject(video_object, object_prefixes):
        # first, base decision on name prefix ..
        for object_prefix in object_prefixes:
            if object_prefix.lower() == video_object.id[:len(object_prefix)].lower():
                return True

        # other potential checks ...

        return False

    @staticmethod
    def generate_XML_objects(filepath, frame_width, frame_height, polygons):
        annotation = ET.Element('annotation')

        # ... image size information ...
        size = ET.SubElement(annotation, 'size')
        width = ET.SubElement(size, 'width')
        width.text = str(frame_width)
        height = ET.SubElement(size, 'height')
        height.text = str(frame_height)
        depth = ET.SubElement(size, 'depth')
        depth.text = str(3)

        # ... image location ...
        folder_name, image_filename = os.path.split(filepath)
        filename = ET.SubElement(annotation, 'filename')
        filename.text = image_filename
        folder = ET.SubElement(annotation, 'folder')
        folder.text = folder_name

        # ... object bboxes...
        for object_name, polygon in polygons:
            obj = ET.SubElement(annotation, 'object')
            name = ET.SubElement(obj, 'name')
            name.text = 'text'
            objID = ET.SubElement(obj, 'ID')
            objID.text = object_name

            polygon_xml = ET.SubElement(obj, 'polygon')
            for p_idx, (px, py) in enumerate(polygon):
                px_xml = ET.SubElement(polygon_xml, 'x' + str(p_idx))
                px_xml.text = str(px)
                py_xml = ET.SubElement(polygon_xml, 'y' + str(p_idx))
                py_xml.text = str(py)

        annotation = ET.ElementTree(annotation)

        return annotation

    @staticmethod
    def FromAnnotationXML(export_mode, export_prefixes, export_speaker_name, export_max_speaker_intersection,
                          output_dir, database, lecture, export_dir, export_images=False):
        # Load video annotations ....
        # ... file name ...
        annotation_prefix = output_dir + "/" + database.output_annotations
        annotation_suffix = database.name + "_" + lecture.title.lower() + ".xml"

        input_main_file = annotation_prefix + "/" + annotation_suffix

        annotation = LectureAnnotation.Load(input_main_file, True)

        text_exporter = TextAnnotationExporter(annotation, export_prefixes, export_speaker_name,
                                               export_max_speaker_intersection, export_mode, export_dir,
                                               export_images=export_images)

        return text_exporter


