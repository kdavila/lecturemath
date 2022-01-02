
import xml.etree.ElementTree as ET

import cv2
import numpy as np

class KeyFrameProjection:
    def __init__(self, w, h, src_points, base_dst_points, delta_x, delta_y, matrix, inv_matrix=None):
        self.width = w
        self.height = h
        self.src_points = src_points
        self.base_dst_points = base_dst_points
        # self.dst_points = []
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.H = matrix
        if inv_matrix is None:
            self.inv_H = np.linalg.inv(self.H)
        else:
            self.inv_H = inv_matrix

    def copy(self):
        return KeyFrameProjection(self.width, self.height, self.src_points.copy(), self.base_dst_points.copy(),
                                  self.delta_x, self.delta_y, self.H.copy(),inv_matrix=self.inv_H.copy())

    def update(self, src_points, base_dst_points, H, delta_x, delta_y):
        self.src_points = src_points.copy()
        self.base_dst_points = base_dst_points.copy()
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.H = H.copy()
        self.inv_H = np.linalg.inv(self.H)

    def warpKeyFrame(self, keyframe, object_mask=False):
        target_size = (self.width, self.height)
        proj_RGB = cv2.warpPerspective(keyframe.raw_image, self.H, target_size)
        proj_BIN = cv2.warpPerspective(keyframe.binary_image, self.H, target_size,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        if object_mask:
            # assuming that mask should be a bool array ... convert to uint8 and make it 255 for True
            tempo_mask = keyframe.object_mask.astype(np.uint8) * 255
            # ... warping ...
            proj_MASK = cv2.warpPerspective(tempo_mask, self.H, target_size)
            # ... back  to boolean ...
            proj_MASK = proj_MASK > 0

            return proj_RGB, proj_BIN, proj_MASK
        else:
            return proj_RGB, proj_BIN

    def warpImage(self, image, inverse=False):
        target_size = (self.width, self.height)

        if inverse:
            warped_image = cv2.warpPerspective(image, self.inv_H, target_size)
        else:
            warped_image = cv2.warpPerspective(image, self.H, target_size)

        return warped_image


    def bboxesToPolygons(self, bboxes):
        polygons = []
        for x, y, w, h in bboxes:
            polygon_points = np.array([
                [x, y], [x + w, y], [x + w, y + h], [x, y + h]
            ], dtype=np.float64)
            polygons.append(polygon_points)

        return polygons

    def warpPoint(self, click_x, click_y, invert=False):
        tempo_point_array = np.array([[[click_x, click_y]]], dtype=np.float32)
        if invert:
            warped_point_array = cv2.perspectiveTransform(tempo_point_array, self.inv_H).reshape(-1, 2)
        else:
            warped_point_array = cv2.perspectiveTransform(tempo_point_array, self.H).reshape(-1, 2)

        warped_x, warped_y = warped_point_array[0]

        return warped_x, warped_y


    def warpPolygon(self, polygon, invert=False):
        if invert:
            return cv2.perspectiveTransform(polygon.reshape(-1, 1, 2), self.inv_H).reshape(-1, 2)
        else:
            return cv2.perspectiveTransform(polygon.reshape(-1, 1, 2), self.H).reshape(-1, 2)

    def warpPolygons(self, polygons, invert=False):
        warped_polygons = []
        for polygon in polygons:
            projected = self.warpPolygon(polygon, invert)

            warped_polygons.append(projected)

        return warped_polygons

    def GenerateXML(self):
        xml_str = "  <KeyFrameProjection>\n"
        xml_str += "    <Width>" + str(self.width) + "</Width>\n"
        xml_str += "    <Height>" + str(self.height) + "</Height>\n"
        xml_str += "    <SourcePoints>\n"
        for idx in range(self.src_points.shape[0]):
            xml_str += "        <Point>\n"
            xml_str += "            <X>" + str(self.src_points[idx, 0]) + "</X>\n"
            xml_str += "            <Y>" + str(self.src_points[idx, 1]) + "</Y>\n"
            xml_str += "        </Point>\n"
        xml_str += "    </SourcePoints>\n"
        xml_str += "    <BaseDestinationPoints>\n"
        for idx in range(self.base_dst_points.shape[0]):
            xml_str += "        <Point>\n"
            xml_str += "            <X>" + str(self.base_dst_points[idx, 0]) + "</X>\n"
            xml_str += "            <Y>" + str(self.base_dst_points[idx, 1]) + "</Y>\n"
            xml_str += "        </Point>\n"
        xml_str += "    </BaseDestinationPoints>\n"
        xml_str += "    <DeltaX>" + str(self.delta_x) + "</DeltaX>\n"
        xml_str += "    <DeltaY>" + str(self.delta_y) + "</DeltaY>\n"
        xml_str += "    <Projection>\n"
        for row_idx in range(self.H.shape[0]):
            for col_idx in range(self.H.shape[0]):
                position_tag = "Value_{0:d}_{1:d}".format(row_idx, col_idx)
                xml_str += "       <{0:s}>{1:s}</{0:s}>\n".format(position_tag, str(self.H[row_idx][col_idx]))
        xml_str += "    </Projection>\n"
        xml_str += "  </KeyFrameProjection>\n"

        return xml_str

    @staticmethod
    def CreateDefault(w, h, offset=10.0):
        points = [[offset, offset], [w - offset, offset], [w - offset, h - offset], [offset, h - offset]]

        default_polygon = np.array(points, dtype=np.float64)
        default_projection = np.identity(3, dtype=np.float64)

        return KeyFrameProjection(w, h, default_polygon.copy(), default_polygon.copy(), 0, 0, default_projection)

    @staticmethod
    def LoadPolygonFromXML(root, namespace):
        loaded_points = []

        xml_points = root.findall(namespace + 'Point')
        for xml_point in xml_points:
            x = float(xml_point.find(namespace + 'X').text)
            y = float(xml_point.find(namespace + 'Y').text)
            loaded_points.append([x, y])

        return np.array(loaded_points, dtype=np.float64)

    @staticmethod
    def LoadKeyFrameProjectionFromXML(root, namespace):
        w = int(root.find(namespace + 'Width').text)
        h = int(root.find(namespace + 'Height').text)

        xml_source_points = root.find(namespace + 'SourcePoints')
        src_points = KeyFrameProjection.LoadPolygonFromXML(xml_source_points, namespace)

        xml_dst_points = root.find(namespace + 'BaseDestinationPoints')
        dst_points = KeyFrameProjection.LoadPolygonFromXML(xml_dst_points, namespace)

        dx = int(root.find(namespace + 'DeltaX').text)
        dy = int(root.find(namespace + 'DeltaY').text)

        xml_projection = root.find(namespace + 'Projection')
        matrix = np.zeros((3, 3), dtype=np.float64)
        for row_idx in range(3):
            for col_idx in range(3):
                value_tag = 'Value_{0:d}_{1:d}'.format(row_idx, col_idx)
                matrix[row_idx, col_idx] = float(xml_projection.find(namespace + value_tag).text)

        return KeyFrameProjection(w, h, src_points, dst_points, dx, dy, matrix)

    @staticmethod
    def GenerateKeyFramesProjectionsXML(all_projections):
        xml_str = " <VideoKeyFramesProjections>\n"
        for kf_projection in all_projections:
            xml_str += kf_projection.GenerateXML()
        xml_str += " </VideoKeyFramesProjections>\n"

        return xml_str

    @staticmethod
    def LoadKeyFramesProjectionsFromXML(xml_filename, namespace):
        tree = ET.parse(xml_filename)
        root = tree.getroot() # ProjectionAnnotations

        projections_root = root.find(namespace + 'VideoKeyFramesProjections')
        all_projections_xml_roots = projections_root.findall(namespace + 'KeyFrameProjection')

        all_projections = []
        for kf_xml_root in all_projections_xml_roots:
            kf_projection = KeyFrameProjection.LoadKeyFrameProjectionFromXML(kf_xml_root, namespace)
            all_projections.append(kf_projection)

        return all_projections

