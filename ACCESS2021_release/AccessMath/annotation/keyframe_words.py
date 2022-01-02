
import xml.etree.ElementTree as ET

from AccessMath.annotation.keyframe_annotation import KeyFrameAnnotation
from AccessMath.annotation.keyframe_projection import KeyFrameProjection
from AccessMath.preprocessing.content.segmentation_tree import SegmentationTree

class KeyFrameWords:
    def __init__(self, kf_annotation, kf_projection, segment_tree):
        self.kf_annotation = kf_annotation
        self.projection = kf_projection
        self.segment_tree = segment_tree

    def getWarpedKeyFrame(self):
        return self.projection.warpKeyFrame(self.kf_annotation)

    def get_words(self):
        return self.segment_tree.collect_all_leaves()

    def words_in_region(self, min_x, max_x, min_y, max_y):
        all_words = self.get_words()

        in_region = []
        for (b_x, b_y, b_w, b_h) in all_words:
            if min_x <= b_x and b_x + b_w <= max_x and min_y <= b_y and b_y + b_h <= max_y:
                in_region.append((b_x, b_y, b_w, b_h))

        return in_region

    def GenerateXML(self):
        xml_str = " <KeyFrameWords>\n"
        xml_str += self.projection.GenerateXML()
        xml_str += self.segment_tree.to_xml()
        xml_str += " </KeyFrameWords>\n"

        return xml_str

    @staticmethod
    def CreateDefault(kf_annotation, proj_offset=10.0):
        raw_h, raw_w, _ = kf_annotation.raw_image.shape
        inv_binary = 255 - kf_annotation.binary_image
        def_segment = SegmentationTree.CreateDefault(inv_binary)
        def_proj = KeyFrameProjection.CreateDefault(raw_w, raw_h, proj_offset)
        def_words = KeyFrameWords(kf_annotation, def_proj, def_segment)

        return def_words

    @staticmethod
    def LoadFromXML(xml_root, namespace, kf_annotation):
        # First, load projection ...
        projection_root = xml_root.find(namespace + 'KeyFrameProjection')
        projection = KeyFrameProjection.LoadKeyFrameProjectionFromXML(projection_root, namespace)

        # get projected binary image used for current segmentation tree
        _, proj_BIN = projection.warpKeyFrame(kf_annotation)
        proj_inv_BIN = 255 - proj_BIN[:, :, 0]

        # now load the corresponding segmentation tree ...
        segmentation_root = xml_root.find(namespace + 'SegmentationTree')
        segmentation = SegmentationTree.from_xml(segmentation_root, proj_inv_BIN)

        return KeyFrameWords(kf_annotation, projection, segmentation)

    @staticmethod
    def LoadKeyFramesWordsFromXML(xml_filename, keyframe_annotations, namespace=''):
        tree = ET.parse(xml_filename)
        root = tree.getroot()

        kf_words_root = root.find(namespace + 'VideoKeyFramesWords')
        all_kf_words_xml_roots = kf_words_root.findall(namespace + 'KeyFrameWords')

        all_kf_words = []
        for kf_idx, kf_words_xml_root in enumerate(all_kf_words_xml_roots):
            kf_words = KeyFrameWords.LoadFromXML(kf_words_xml_root, namespace, keyframe_annotations[kf_idx])
            all_kf_words.append(kf_words)

        return all_kf_words

    @staticmethod
    def KeyFramesWordsToXML(video_kf_words):
        xml_str = " <VideoKeyFramesWords>\n"
        for kf_words in video_kf_words:
            xml_str += kf_words.GenerateXML()
        xml_str += " </VideoKeyFramesWords>\n"

        return xml_str

