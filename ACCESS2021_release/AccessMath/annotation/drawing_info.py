
import numpy as np

class DrawingInfo:
    def __init__(self, canvas_bbox, player_control_bbox, player_render_bbox):
        self.canvas_bbox = canvas_bbox
        self.player_control_bbox = player_control_bbox
        self.player_render_bbox = player_render_bbox

        self.proj_off_x = self.player_render_bbox[0] - self.canvas_bbox[0]
        self.proj_off_y = self.player_render_bbox[1] - self.canvas_bbox[1]
        self.proj_off_m = np.array([[self.proj_off_x, self.proj_off_y]])

    def equivalent_bboxes(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        return (int(round(x1)) == int(round(x2)) and int(round(y1)) == int(round(y2)) and
                int(round(w1)) == int(round(w2)) and int(round(h1)) == int(round(h2)))

    def equivalent_bboxes_area(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        return (int(round(w1)) == int(round(w2)) and int(round(h1)) == int(round(h2)))

    def canvas_render_dist(self):
        x1, y1, w1, h1 = self.canvas_bbox
        x2, y2, w2, h2 = self.player_render_bbox

        return (x2 - x1, y2 - y1)

    def equivalent_areas(self, other):
        if isinstance(other, DrawingInfo):
            return (self.equivalent_bboxes_area(self.canvas_bbox, other.canvas_bbox) and
                    self.equivalent_bboxes_area(self.player_control_bbox, other.player_control_bbox) and
                    self.equivalent_bboxes_area(self.player_render_bbox, other.player_render_bbox) and
                    self.canvas_render_dist() == other.canvas_render_dist())
        else:
            return False

    def __eq__(self, other):
        if isinstance(other, DrawingInfo):
            return (self.equivalent_bboxes(self.canvas_bbox, other.canvas_bbox) and
                    self.equivalent_bboxes(self.player_control_bbox, other.player_control_bbox) and
                    self.equivalent_bboxes(self.player_render_bbox, other.player_render_bbox))
        else:
            return False

    def __str__(self):
        canvas_str = str(self.canvas_bbox)
        control_str = str(self.player_control_bbox)
        render_str = str(self.player_render_bbox)
        return "<DrawInfo: Canvas={0}, Control={1}, Render={2}>".format(canvas_str, control_str, render_str)

    def project_polygon(self, img_width, img_height, polygon_points):
        # note that these values should be the same (if aspect ratio is kept)
        proj_scale_x = img_width / self.player_render_bbox[2]
        proj_scale_y = img_height / self.player_render_bbox[3]

        proj_points = (polygon_points - self.proj_off_m)
        proj_points[:, 0] *= proj_scale_x
        proj_points[:, 1] *= proj_scale_y

        return proj_points

    def unproject_polygon(self, img_width, img_height, polygon_points):
        # note that these values should be the same (if aspect ratio is kept)
        unproj_scale_w = self.player_render_bbox[2] / img_width
        unproj_scale_h = self.player_render_bbox[3] / img_height

        # correct scale first ...
        unproj_points = polygon_points.copy()
        unproj_points[:, 0] *= unproj_scale_w
        unproj_points[:, 1] *= unproj_scale_h
        # then translate ...
        unproj_points = (unproj_points + self.proj_off_m)

        return unproj_points

    def generate_xml(self):
        canvas_x, canvas_y, canvas_w, canvas_h = self.canvas_bbox
        player_x, player_y, player_w, player_h = self.player_control_bbox
        render_x, render_y, render_w, render_h = self.player_render_bbox

        xml_string = "  <DrawingInfo>\n"
        xml_string += "     <Canvas>\n"
        xml_string += "         <X>" + str(canvas_x) + "</X>\n"
        xml_string += "         <Y>" + str(canvas_y) + "</Y>\n"
        xml_string += "         <W>" + str(canvas_w) + "</W>\n"
        xml_string += "         <H>" + str(canvas_h) + "</H>\n"
        xml_string += "     </Canvas>\n"
        xml_string += "     <Player>\n"
        xml_string += "         <ControlArea>\n"
        xml_string += "             <X>" + str(player_x) + "</X>\n"
        xml_string += "             <Y>" + str(player_y) + "</Y>\n"
        xml_string += "             <W>" + str(player_w) + "</W>\n"
        xml_string += "             <H>" + str(player_h) + "</H>\n"
        xml_string += "         </ControlArea>\n"
        xml_string += "         <RenderArea>\n"
        xml_string += "             <X>" + str(render_x) + "</X>\n"
        xml_string += "             <Y>" + str(render_y) + "</Y>\n"
        xml_string += "             <W>" + str(render_w) + "</W>\n"
        xml_string += "             <H>" + str(render_h) + "</H>\n"
        xml_string += "         </RenderArea>\n"
        xml_string += "     </Player>\n"
        xml_string += "  </DrawingInfo>\n"

        return xml_string

    @staticmethod
    def load_bbox_from_XML(root, namespace):
        x = float(root.find(namespace + "X").text)
        y = float(root.find(namespace + "Y").text)
        w = float(root.find(namespace + "W").text)
        h = float(root.find(namespace + "H").text)

        return [x, y, w, h]

    @staticmethod
    def from_XML(root, namespace):
        draw_info_root = root.find(namespace + "DrawingInfo")

        canvas_bbox_root = draw_info_root.find(namespace + "Canvas")
        canvas_bbox = DrawingInfo.load_bbox_from_XML(canvas_bbox_root, namespace)

        player_root = draw_info_root.find(namespace + "Player")
        control_bbox_root = player_root.find(namespace + "ControlArea")
        control_bbox = DrawingInfo.load_bbox_from_XML(control_bbox_root, namespace)

        render_bbox_root = player_root.find(namespace + "RenderArea")
        render_bbox = DrawingInfo.load_bbox_from_XML(render_bbox_root, namespace)

        draw_info = DrawingInfo(canvas_bbox, control_bbox, render_bbox)

        return draw_info



