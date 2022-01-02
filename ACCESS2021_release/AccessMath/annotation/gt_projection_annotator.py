
import os

import cv2
import numpy as np

from AM_CommonTools.util.time_helper import TimeHelper
from AM_CommonTools.interface.controls.screen import Screen
from AM_CommonTools.interface.controls.screen_button import ScreenButton
from AM_CommonTools.interface.controls.screen_canvas import ScreenCanvas
from AM_CommonTools.interface.controls.screen_container import ScreenContainer
from AM_CommonTools.interface.controls.screen_horizontal_scroll import ScreenHorizontalScroll
from AM_CommonTools.interface.controls.screen_image import ScreenImage
from AM_CommonTools.interface.controls.screen_label import ScreenLabel

from AccessMath.annotation.keyframe_annotation import KeyFrameAnnotation
from AccessMath.annotation.keyframe_projection import KeyFrameProjection


class GTProjectionAnnotator(Screen):
    ModeNavigate = 0
    ModeProjectionRegionSelection = 1
    ModeProjectionRegionAdjustment = 2
    ModeExitConfirm = 3

    ViewModeNormalRGB = 0
    ViewModeProjectedRGB = 1
    ViewModeNormalBin = 2
    ViewModeProjectedBin = 3

    ParamsMaxProjectionTraslation = 250

    def __init__(self, size, db_name, lecture_title, output_path):
        Screen.__init__(self, "Projection Ground Truth Annotation Interface", size)

        general_background = (100, 90, 80)
        darker_background = (50, 45, 40)
        text_color = (255, 255, 255)
        button_text_color = (35, 50, 20)
        button_back_color = (228, 228, 228)
        self.elements.back_color = general_background

        self.db_name = db_name
        self.lecture_title = lecture_title

        self.output_path = output_path

        segments_filename = self.output_path + "/segments.xml"
        keyframes_image_prefix = self.output_path + "/keyframes/"

        # load including segment information
        # .... key-frames will not be combined in this mode
        self.keyframe_annotations, self.segments = KeyFrameAnnotation.LoadExportedKeyframes(segments_filename,
                                                                                            keyframes_image_prefix,
                                                                                            True)

        if len(self.keyframe_annotations) > 0:
            print("Key-frames Loaded: " + str(len(self.keyframe_annotations)))
        else:
            raise Exception("Cannot start with 0 key-frames")

        self.projected_RGB_cache = {}
        self.projected_BIN_cache = {}

        # Loading data specific to the Projection Annotation process ...
        self.kf_projections = []

        # .... try loading from file ....
        proj_filename = self.output_path + "/projections.xml"

        if os.path.exists(proj_filename):
            print("Loading saved projection annotations")
            self.kf_projections = KeyFrameProjection.LoadKeyFramesProjectionsFromXML(proj_filename, "")
        else:
            print("No previous Projection annotations found")
            # ... no data found? create defaults ...
            for idx in range(len(self.keyframe_annotations)):
                raw_h, raw_w, _ = self.keyframe_annotations[idx].raw_image.shape
                self.kf_projections.append(KeyFrameProjection.CreateDefault(raw_w, raw_h, 10.0))

        # view caches ...
        self.update_projected_cache()

        # Creating interface ...
        self.view_mode = GTProjectionAnnotator.ViewModeNormalRGB
        self.edition_mode = GTProjectionAnnotator.ModeNavigate
        self.view_scale = 1.0
        self.selected_keyframe = 0

        self.base_projection = None
        self.projection_src_points = None
        self.projection_base_dst_points = None
        self.projection_dst_points = None
        self.projection_H = None
        # self.projection_H_inv = None
        self.projection_delta_x = 0
        self.projection_delta_y = 0

        # add elements....
        container_top = 10
        container_width = 330

        button_2_width = 150
        button_2_left = int(container_width * 0.25) - button_2_width / 2
        button_2_right = int(container_width * 0.75) - button_2_width / 2

        # ======================================================================
        #   Navigation panel to move accross frames
        self.container_nav_buttons = ScreenContainer("container_nav_buttons", (container_width, 70),
                                                     back_color=general_background)
        self.container_nav_buttons.position = (self.width - self.container_nav_buttons.width - 10, container_top)
        self.elements.append(self.container_nav_buttons)

        self.lbl_nav_keyframe = ScreenLabel("lbl_nav_keyframe", "Key-Frame: 1 / " + str(len(self.keyframe_annotations)),
                                            21, 290, 1)
        self.lbl_nav_keyframe.position = (5, 5)
        self.lbl_nav_keyframe.set_background(general_background)
        self.lbl_nav_keyframe.set_color(text_color)
        self.container_nav_buttons.append(self.lbl_nav_keyframe)

        time_str = TimeHelper.stampToStr(self.keyframe_annotations[self.selected_keyframe].time)
        self.lbl_nav_time = ScreenLabel("lbl_nav_time", time_str, 21, 290, 1)
        self.lbl_nav_time.position = (5, self.lbl_nav_keyframe.get_bottom() + 20)
        self.lbl_nav_time.set_background(general_background)
        self.lbl_nav_time.set_color(text_color)
        self.container_nav_buttons.append(self.lbl_nav_time)

        self.btn_nav_keyframe_prev = ScreenButton("btn_nav_keyframe_prev", "Prev", 21, 90)
        self.btn_nav_keyframe_prev.set_colors(button_text_color, button_back_color)
        self.btn_nav_keyframe_prev.position = (10, self.lbl_nav_keyframe.get_bottom() + 10)
        self.btn_nav_keyframe_prev.click_callback = self.btn_nav_keyframe_prev_click
        self.container_nav_buttons.append(self.btn_nav_keyframe_prev)

        self.btn_nav_keyframe_next = ScreenButton("btn_nav_keyframe_next", "Next", 21, 90)
        self.btn_nav_keyframe_next.set_colors(button_text_color, button_back_color)
        self.btn_nav_keyframe_next.position = (self.container_nav_buttons.width - self.btn_nav_keyframe_next.width - 10,
                                               self.lbl_nav_keyframe.get_bottom() + 10)
        self.btn_nav_keyframe_next.click_callback = self.btn_nav_keyframe_next_click
        self.container_nav_buttons.append(self.btn_nav_keyframe_next)

        # ======================================================================
        # confirmation panel
        self.container_confirm_buttons = ScreenContainer("container_confirm_buttons", (container_width, 70),
                                                         back_color=general_background)
        self.container_confirm_buttons.position = (
        self.width - self.container_confirm_buttons.width - 10, container_top)
        self.elements.append(self.container_confirm_buttons)
        self.container_confirm_buttons.visible = False

        self.lbl_confirm_message = ScreenLabel("lbl_confirm_message", "Confirmation message goes here?", 21, 290, 1)
        self.lbl_confirm_message.position = (5, 5)
        self.lbl_confirm_message.set_background(general_background)
        self.lbl_confirm_message.set_color(text_color)
        self.container_confirm_buttons.append(self.lbl_confirm_message)

        self.btn_confirm_cancel = ScreenButton("btn_confirm_cancel", "Cancel", 21, 130)
        self.btn_confirm_cancel.set_colors(button_text_color, button_back_color)
        self.btn_confirm_cancel.position = (10, self.lbl_nav_keyframe.get_bottom() + 10)
        self.btn_confirm_cancel.click_callback = self.btn_confirm_cancel_click
        self.container_confirm_buttons.append(self.btn_confirm_cancel)

        self.btn_confirm_accept = ScreenButton("btn_confirm_accept", "Accept", 21, 130)
        self.btn_confirm_accept.set_colors(button_text_color, button_back_color)
        self.btn_confirm_accept.position = (self.container_confirm_buttons.width - self.btn_confirm_accept.width - 10,
                                            self.lbl_confirm_message.get_bottom() + 10)
        self.btn_confirm_accept.click_callback = self.btn_confirm_accept_click
        self.container_confirm_buttons.append(self.btn_confirm_accept)

        # ======================================================================
        # View panel with view control buttons
        self.container_view_buttons = ScreenContainer("container_view_buttons", (container_width, 165),
                                                      back_color=general_background)
        self.container_view_buttons.position = (self.width - self.container_view_buttons.width - 10,
                                                self.container_nav_buttons.get_bottom() + 10)
        self.elements.append(self.container_view_buttons)

        button_width = 190
        button_left = (self.container_view_buttons.width - button_width) / 2

        # zoom ....
        self.lbl_zoom = ScreenLabel("lbl_zoom", "Zoom: 100%", 21, container_width - 10, 1)
        self.lbl_zoom.position = (5, 5)
        self.lbl_zoom.set_background(general_background)
        self.lbl_zoom.set_color(text_color)
        self.container_view_buttons.append(self.lbl_zoom)

        self.btn_zoom_reduce = ScreenButton("btn_zoom_reduce", "[ - ]", 21, 90)
        self.btn_zoom_reduce.set_colors(button_text_color, button_back_color)
        self.btn_zoom_reduce.position = (10, self.lbl_zoom.get_bottom() + 10)
        self.btn_zoom_reduce.click_callback = self.btn_zoom_reduce_click
        self.container_view_buttons.append(self.btn_zoom_reduce)

        self.btn_zoom_increase = ScreenButton("btn_zoom_increase", "[ + ]", 21, 90)
        self.btn_zoom_increase.set_colors(button_text_color, button_back_color)
        self.btn_zoom_increase.position = (self.container_view_buttons.width - self.btn_zoom_increase.width - 10,
                                           self.lbl_zoom.get_bottom() + 10)
        self.btn_zoom_increase.click_callback = self.btn_zoom_increase_click
        self.container_view_buttons.append(self.btn_zoom_increase)

        self.btn_zoom_zero = ScreenButton("btn_zoom_zero", "100%", 21, 90)
        self.btn_zoom_zero.set_colors(button_text_color, button_back_color)
        self.btn_zoom_zero.position = ((self.container_view_buttons.width - self.btn_zoom_zero.width) / 2,
                                       self.lbl_zoom.get_bottom() + 10)
        self.btn_zoom_zero.click_callback = self.btn_zoom_zero_click
        self.container_view_buttons.append(self.btn_zoom_zero)

        self.btn_view_normal_rgb = ScreenButton("btn_view_normal_rgb", "Normal RGB", 21, button_2_width)
        self.btn_view_normal_rgb.set_colors(button_text_color, button_back_color)
        self.btn_view_normal_rgb.position = (button_2_left, self.btn_zoom_zero.get_bottom() + 10)
        self.btn_view_normal_rgb.click_callback = self.btn_view_normal_rgb_click
        self.container_view_buttons.append(self.btn_view_normal_rgb)

        self.btn_view_projected_rgb = ScreenButton("btn_view_projected_rgb", "Projected RGB", 21, button_2_width)
        self.btn_view_projected_rgb.set_colors(button_text_color, button_back_color)
        self.btn_view_projected_rgb.position = (button_2_right, self.btn_zoom_zero.get_bottom() + 10)
        self.btn_view_projected_rgb.click_callback = self.btn_view_projected_rgb_click
        self.container_view_buttons.append(self.btn_view_projected_rgb)

        self.btn_view_normal_bin = ScreenButton("btn_view_normal_bin", "Normal BIN", 21, button_2_width)
        self.btn_view_normal_bin.set_colors(button_text_color, button_back_color)
        self.btn_view_normal_bin.position =  (button_2_left, self.btn_view_normal_rgb.get_bottom() + 10)
        self.btn_view_normal_bin.click_callback = self.btn_view_normal_bin_click
        self.container_view_buttons.append(self.btn_view_normal_bin)

        self.btn_view_projected_bin = ScreenButton("btn_view_projected_bin", "Projected BIN", 21, button_2_width)
        self.btn_view_projected_bin.set_colors(button_text_color, button_back_color)
        self.btn_view_projected_bin.position = (button_2_right, self.btn_view_normal_bin.get_top())
        self.btn_view_projected_bin.click_callback = self.btn_view_projected_bin_click
        self.container_view_buttons.append(self.btn_view_projected_bin)

        self.btn_view_normal_bin.visible = False
        self.btn_view_projected_bin.visible = False

        # ======================================================================
        # Projection Adjustment Panel
        self.container_projection_buttons = ScreenContainer("container_projection_buttons", (container_width, 95),
                                                            darker_background)
        self.container_projection_buttons.position = (self.container_view_buttons.get_left(),
                                                      self.container_view_buttons.get_bottom() + 5)
        self.elements.append(self.container_projection_buttons)

        self.btn_projection_edit = ScreenButton("btn_projection_edit", "Edit Projection", 21, button_width)
        self.btn_projection_edit.set_colors(button_text_color, button_back_color)
        self.btn_projection_edit.position = (button_left, 5)
        self.btn_projection_edit.click_callback = self.btn_projection_edit_click
        self.container_projection_buttons.append(self.btn_projection_edit)

        self.btn_projection_copy_prev = ScreenButton("btn_projection_copy_prev", "Copy Prev", 21, button_2_width)
        self.btn_projection_copy_prev.set_colors(button_text_color, button_back_color)
        self.btn_projection_copy_prev.position = (button_2_left, self.btn_projection_edit.get_bottom() + 10)
        self.btn_projection_copy_prev.click_callback = self.btn_projection_copy_prev_click
        self.container_projection_buttons.append(self.btn_projection_copy_prev)

        self.btn_projection_copy_next = ScreenButton("btn_projection_copy_next", "Copy Next", 21, button_2_width)
        self.btn_projection_copy_next.set_colors(button_text_color, button_back_color)
        self.btn_projection_copy_next.position = (button_2_right, self.btn_projection_edit.get_bottom() + 10)
        self.btn_projection_copy_next.click_callback = self.btn_projection_copy_next_click
        self.container_projection_buttons.append(self.btn_projection_copy_next)


        # ===============================================

        # Panel with Projection parameters for step 2 (Adjusting Projection)
        self.container_projection_adjust = ScreenContainer("container_projection_adjust", (container_width, 150),
                                                              general_background)
        self.container_projection_adjust.position = (self.container_view_buttons.get_left(),
                                                     self.container_view_buttons.get_bottom() + 5)
        self.elements.append(self.container_projection_adjust)

        self.lbl_translation_title = ScreenLabel("lbl_translation_title", "Translation Parameters", 21,
                                                 container_width - 10, 1)
        self.lbl_translation_title.position = (5, 5)
        self.lbl_translation_title.set_background(general_background)
        self.lbl_translation_title.set_color(text_color)
        self.container_projection_adjust.append(self.lbl_translation_title)

        self.lbl_delta_x = ScreenLabel("lbl_delta_x", "Delta X: " + str(self.projection_delta_x), 21, container_width - 10, 1)
        self.lbl_delta_x.position = (5, self.lbl_translation_title.get_bottom() + 20)
        self.lbl_delta_x.set_background(general_background)
        self.lbl_delta_x.set_color(text_color)
        self.container_projection_adjust.append(self.lbl_delta_x)

        max_delta = GTProjectionAnnotator.ParamsMaxProjectionTraslation
        self.scroll_delta_x = ScreenHorizontalScroll("scroll_delta_x", -max_delta, max_delta, 0, 1)
        self.scroll_delta_x.position = (5, self.lbl_delta_x.get_bottom() + 10)
        self.scroll_delta_x.width = container_width - 10
        self.scroll_delta_x.scroll_callback = self.scroll_delta_x_change
        self.container_projection_adjust.append(self.scroll_delta_x)

        self.lbl_delta_y = ScreenLabel("lbl_delta_y", "Delta Y: " + str(self.projection_delta_y), 21, container_width - 10, 1)
        self.lbl_delta_y.position = (5, self.scroll_delta_x.get_bottom() + 20)
        self.lbl_delta_y.set_background(general_background)
        self.lbl_delta_y.set_color(text_color)
        self.container_projection_adjust.append(self.lbl_delta_y)

        self.scroll_delta_y = ScreenHorizontalScroll("scroll_delta_y", -max_delta, max_delta, 0, 1)
        self.scroll_delta_y.position = (5, self.lbl_delta_y.get_bottom() + 10)
        self.scroll_delta_y.width = container_width - 10
        self.scroll_delta_y.scroll_callback = self.scroll_delta_y_change
        self.container_projection_adjust.append(self.scroll_delta_y)

        self.container_projection_adjust.visible = False

        #=============================================================
        # Panel with state buttons (Undo, Redo, Save, Export)
        self.container_state_buttons = ScreenContainer("container_state_buttons", (container_width, 250),
                                                       general_background)
        self.container_state_buttons.position = (
        self.container_view_buttons.get_left(), self.container_projection_buttons.get_bottom() + 10)
        self.elements.append(self.container_state_buttons)

        self.btn_undo = ScreenButton("btn_undo", "Undo", 21, button_width)
        self.btn_undo.set_colors(button_text_color, button_back_color)
        self.btn_undo.position = (button_left, 5)
        self.btn_undo.click_callback = self.btn_undo_click
        self.container_state_buttons.append(self.btn_undo)

        self.btn_redo = ScreenButton("btn_redo", "Redo", 21, button_width)
        self.btn_redo.set_colors(button_text_color, button_back_color)
        self.btn_redo.position = (button_left, self.btn_undo.get_bottom() + 10)
        self.btn_redo.click_callback = self.btn_redo_click
        self.container_state_buttons.append(self.btn_redo)

        self.btn_save = ScreenButton("btn_save", "Save", 21, button_width)
        self.btn_save.set_colors(button_text_color, button_back_color)
        self.btn_save.position = (button_left, self.btn_redo.get_bottom() + 10)
        self.btn_save.click_callback = self.btn_save_click
        self.container_state_buttons.append(self.btn_save)

        self.btn_exit = ScreenButton("btn_exit", "Exit", 21, button_width)
        self.btn_exit.set_colors(button_text_color, button_back_color)
        self.btn_exit.position = (button_left, self.btn_save.get_bottom() + 30)
        self.btn_exit.click_callback = self.btn_exit_click
        self.container_state_buttons.append(self.btn_exit)

        # ==============================================================
        # Image Viewer and Canvas

        image_width = self.width - self.container_nav_buttons.width - 30
        image_height = self.height - container_top - 10
        self.container_images = ScreenContainer("container_images", (image_width, image_height), back_color=(0, 0, 0))
        self.container_images.position = (10, container_top)
        self.elements.append(self.container_images)

        # ... image objects ...
        tempo_blank = np.zeros((50, 50, 3), np.uint8)
        tempo_blank[:, :, :] = 255
        self.img_main = ScreenImage("img_main", tempo_blank, 0, 0, True, cv2.INTER_NEAREST)
        self.img_main.position = (0, 0)
        self.img_main.mouse_button_down_callback = self.img_mouse_down
        self.container_images.append(self.img_main)

        # canvas used for annotations
        self.canvas_select = ScreenCanvas("canvas_select", 100, 100)
        self.canvas_select.position = (0, 0)
        self.canvas_select.locked = True
        # self.canvas_select.object_edited_callback = self.canvas_object_edited
        # self.canvas_select.object_selected_callback = self.canvas_selection_changed
        self.container_images.append(self.canvas_select)

        self.canvas_select.add_polygon_element("selection_polygon", self.kf_projections[0].base_dst_points)
        self.canvas_select.elements["selection_polygon"].visible = False

        self.undo_stack = []
        self.redo_stack = []

        self.elements.key_up_callback = self.main_key_up

        self.update_current_view(True)


    def update_keyframe_projections(self, kf_idx):
        proj_RGB, proj_BIN = self.kf_projections[kf_idx].warpKeyFrame(self.keyframe_annotations[kf_idx])

        self.projected_RGB_cache[kf_idx] = proj_RGB
        self.projected_BIN_cache[kf_idx] = proj_BIN

    def update_projected_cache(self, target_frame=None):
        # compute new frames ...
        if target_frame is None:
            for kf_idx in range(len(self.keyframe_annotations)):
                self.update_keyframe_projections(kf_idx)
        else:
            self.update_keyframe_projections(target_frame)

    def update_current_view(self, resized=False):
        if self.edition_mode == GTProjectionAnnotator.ModeProjectionRegionAdjustment:
            # Force the view in this mode to use the projected image ...
            base_image = self.base_projection
        elif self.view_mode == GTProjectionAnnotator.ViewModeNormalBin:
            base_image = self.keyframe_annotations[self.selected_keyframe].binary_image
        elif self.view_mode == GTProjectionAnnotator.ViewModeProjectedRGB:
            base_image = self.projected_RGB_cache[self.selected_keyframe]
        elif self.view_mode == GTProjectionAnnotator.ViewModeProjectedBin:
            base_image = self.projected_BIN_cache[self.selected_keyframe]
        else:
            # default for ViewModeNormalRGB
            base_image = self.keyframe_annotations[self.selected_keyframe].raw_image

        h, w, c = base_image.shape
        modified_image = base_image.copy()

        # if anything else needs to be displayed ... do it here ....

        # finally, resize ...
        modified_image = cv2.resize(modified_image, (int(w * self.view_scale), int(h * self.view_scale)),
                                    interpolation=cv2.INTER_NEAREST)

        self.canvas_select.height, self.canvas_select.width, _ = modified_image.shape

        # replace/update image
        self.img_main.set_image(modified_image, 0, 0, True, cv2.INTER_NEAREST)
        if resized:
            self.container_images.recalculate_size()

    def update_selected_keyframe(self, new_selected):
        if 0 <= new_selected < len(self.keyframe_annotations):
            self.selected_keyframe = new_selected
        else:
            return

        self.lbl_nav_keyframe.set_text("Key-Frame: " + str(self.selected_keyframe + 1) + " / " +
                                       str(len(self.keyframe_annotations)))

        time_str = TimeHelper.stampToStr(self.keyframe_annotations[self.selected_keyframe].time)
        self.lbl_nav_time.set_text(time_str)

        self.update_current_view()

    def btn_nav_keyframe_next_click(self, button):
        self.update_selected_keyframe(self.selected_keyframe + 1)

    def btn_nav_keyframe_prev_click(self, button):
        self.update_selected_keyframe(self.selected_keyframe - 1)

    def btn_zoom_reduce_click(self, button):
        self.update_view_scale(self.view_scale - 0.25)

    def btn_zoom_increase_click(self, button):
        self.update_view_scale(self.view_scale + 0.25)

    def btn_zoom_zero_click(self, button):
        self.update_view_scale(1.0)

    def update_view_scale(self, new_scale):
        prev_scale = self.view_scale

        if 0.25 <= new_scale <= 4.0:
            self.view_scale = new_scale
        else:
            return

        # keep previous offsets ...
        scroll_offset_y = self.container_images.v_scroll.value if self.container_images.v_scroll.active else 0
        scroll_offset_x = self.container_images.h_scroll.value if self.container_images.h_scroll.active else 0

        prev_center_y = scroll_offset_y + self.container_images.height / 2
        prev_center_x = scroll_offset_x + self.container_images.width / 2

        # compute new scroll bar offsets
        scale_factor = (new_scale / prev_scale)
        new_off_y = prev_center_y * scale_factor - self.container_images.height / 2
        new_off_x = prev_center_x * scale_factor - self.container_images.width / 2

        # update view ....
        self.update_current_view(True)

        # set offsets
        if self.container_images.v_scroll.active and 0 <= new_off_y <= self.container_images.v_scroll.max:
            self.container_images.v_scroll.value = new_off_y
        if self.container_images.h_scroll.active and 0 <= new_off_x <= self.container_images.h_scroll.max:
            self.container_images.h_scroll.value = new_off_x

        selection_polygon = self.canvas_select.elements["selection_polygon"]
        selection_polygon.points *= scale_factor

        # update scale text ...
        self.lbl_zoom.set_text("Zoom: " + str(int(round(self.view_scale * 100,0))) + "%")

    def btn_view_normal_rgb_click(self, button):
        self.view_mode = GTProjectionAnnotator.ViewModeNormalRGB
        self.update_current_view()

    def btn_view_normal_bin_click(self, button):
        self.view_mode = GTProjectionAnnotator.ViewModeNormalBin
        self.update_current_view()

    def btn_view_projected_rgb_click(self, button):
        self.view_mode = GTProjectionAnnotator.ViewModeProjectedRGB
        self.update_current_view()

    def btn_view_projected_bin_click(self, button):
        self.view_mode = GTProjectionAnnotator.ViewModeProjectedBin
        self.update_current_view()

    def img_mouse_down(self, img_object, pos, button):
        pass

    def main_key_up(self, scancode, key, unicode):
        # key short cuts
        if key == 269:
            # minus
            self.btn_zoom_reduce_click(None)
        elif key == 270:
            # plus
            self.btn_zoom_increase_click(None)
        elif key == 110:
            # n
            if self.container_nav_buttons.visible:
                self.btn_nav_keyframe_next_click(None)
        elif key == 112:
            # p
            if self.container_nav_buttons.visible:
                self.btn_nav_keyframe_prev_click(None)

    def set_editor_mode(self, new_mode):
        self.edition_mode = new_mode
        self.container_nav_buttons.visible = (new_mode == GTProjectionAnnotator.ModeNavigate)

        self.container_confirm_buttons.visible = (new_mode == GTProjectionAnnotator.ModeProjectionRegionSelection or
                                                  new_mode == GTProjectionAnnotator.ModeProjectionRegionAdjustment or
                                                  new_mode == GTProjectionAnnotator.ModeExitConfirm)

        if new_mode == GTProjectionAnnotator.ModeProjectionRegionSelection:
            # simulate click ...to normal view ...
            self.btn_view_normal_rgb_click(None)
            self.lbl_confirm_message.set_text("Select a Rectangular Region")
        elif new_mode == GTProjectionAnnotator.ModeProjectionRegionAdjustment:
            self.lbl_confirm_message.set_text("Adjusting Selected Region")
        elif new_mode == GTProjectionAnnotator.ModeExitConfirm:
            self.lbl_confirm_message.set_text("Exit Without Saving?")

        if new_mode == GTProjectionAnnotator.ModeProjectionRegionSelection:
            # show polygon
            self.canvas_select.locked = False
            self.canvas_select.elements["selection_polygon"].visible = True
        else:
            # for every other mode
            self.canvas_select.locked = True
            self.canvas_select.elements["selection_polygon"].visible = False

        self.container_state_buttons.visible = (new_mode == GTProjectionAnnotator.ModeNavigate)

        self.container_projection_buttons.visible = (new_mode == GTProjectionAnnotator.ModeNavigate)
        self.container_projection_adjust.visible = (new_mode == GTProjectionAnnotator.ModeProjectionRegionAdjustment)

        self.btn_view_normal_rgb.visible = (new_mode == GTProjectionAnnotator.ModeNavigate)
        self.btn_view_projected_rgb.visible = (new_mode == GTProjectionAnnotator.ModeNavigate)

    def btn_confirm_cancel_click(self, button):
        # by default, got back to navigation mode ...
        if self.edition_mode == GTProjectionAnnotator.ModeProjectionRegionAdjustment:
            # go back to selection mode
            self.set_editor_mode(GTProjectionAnnotator.ModeProjectionRegionSelection)
        else:
            # all other states
            self.set_editor_mode(GTProjectionAnnotator.ModeNavigate)
        self.update_current_view(False)

    def btn_confirm_accept_click(self, button):

        if self.edition_mode == GTProjectionAnnotator.ModeProjectionRegionSelection:
            # update projection parameters ...
            self.update_projection_image()
            # move to the next stage ..
            self.set_editor_mode(GTProjectionAnnotator.ModeProjectionRegionAdjustment)
            self.update_current_view(False)
        elif self.edition_mode == GTProjectionAnnotator.ModeProjectionRegionAdjustment:
            # update key-frame projection info
            current_projection = self.kf_projections[self.selected_keyframe]
            old_projection =  current_projection.copy()
            current_projection.update(self.projection_src_points, self.projection_base_dst_points,
                                      self.projection_H, self.projection_delta_x, self.projection_delta_y)
            # current_projection.inv_H = self.projection_H_inv

            # update projections ...
            self.update_projected_cache(self.selected_keyframe)

            to_undo = {
                "operation": "projection_changed",
                "keyframe_idx": self.selected_keyframe,
                "old_projection": old_projection,
            }
            self.undo_stack.append(to_undo)

            # update the view ....
            self.set_editor_mode(GTProjectionAnnotator.ModeNavigate)
            self.update_current_view(False)

        elif self.edition_mode == GTProjectionAnnotator.ModeExitConfirm:
            # exit
            self.return_screen = None
            print("Changes have been lost!")
            print("APPLICATION FINISHED")

    def update_projection_image(self):
        self.projection_src_points = self.canvas_select.elements["selection_polygon"].points / self.view_scale

        top_length = np.linalg.norm(self.projection_src_points [1] - self.projection_src_points[0])
        right_length = np.linalg.norm(self.projection_src_points[2] - self.projection_src_points[1])
        bottom_length = np.linalg.norm(self.projection_src_points[3] - self.projection_src_points[2])
        left_length = np.linalg.norm(self.projection_src_points[0] - self.projection_src_points[3])

        target_width = (top_length + bottom_length) / 2.0
        target_height = (left_length + right_length) / 2.0

        target_x = (self.projection_src_points[2, 0] + self.projection_src_points[0, 0] - target_width) / 2
        target_y = (self.projection_src_points[2, 1] + self.projection_src_points[0, 1] - target_height) / 2

        # print((top_length, bottom_length))
        # print((left_length, right_length))
        # print((polygon_points[0, 0], polygon_points[0, 1], (top_length, bottom_length), (left_length, right_length)))
        # print((target_x, target_y, target_width, target_height))

        dst_points = [[target_x, target_y], [target_x + target_width, target_y],
                      [target_x + target_width, target_y + target_height], [target_x, target_y + target_height]]

        self.projection_base_dst_points = np.array(dst_points)

        delta_array = np.array([[self.projection_delta_x, self.projection_delta_y]])
        self.projection_dst_points = self.projection_base_dst_points + delta_array

        self.projection_H, mask = cv2.findHomography(self.projection_src_points, self.projection_dst_points)
        # self.projection_H_inv, _ = cv2.findHomography(self.projection_dst_points, self.projection_src_points)

        current_img = self.keyframe_annotations[self.selected_keyframe].raw_image
        dst_size = (current_img.shape[1], current_img.shape[0])
        self.base_projection = cv2.warpPerspective(current_img, self.projection_H, dst_size)

    def btn_projection_edit_click(self, button):
        # copy existing key-frame parameters to the controllers
        # print(self.selected_keyframe)
        # print(self.kf_projections[self.selected_keyframe].GenerateXML())
        current_proj = self.kf_projections[self.selected_keyframe]
        current_base_points = current_proj.src_points * self.view_scale

        self.canvas_select.update_polygon_element("selection_polygon", current_base_points, True)
        # delta x and y
        self.scroll_delta_x.set_value(current_proj.delta_x)
        self.scroll_delta_y.set_value(current_proj.delta_y)

        self.set_editor_mode(GTProjectionAnnotator.ModeProjectionRegionSelection)
        self.update_current_view(False)

    def btn_projection_copy_next_click(self, button):
        if self.selected_keyframe + 1 < len(self.keyframe_annotations):
            curr_proj = self.kf_projections[self.selected_keyframe].copy()
            proj = self.kf_projections[self.selected_keyframe + 1].copy()
            self.kf_projections[self.selected_keyframe] = proj
            self.update_projected_cache(self.selected_keyframe)

            to_undo = {
                "operation": "projection_changed",
                "keyframe_idx": self.selected_keyframe,
                "old_projection": curr_proj,
            }
            self.undo_stack.append(to_undo)


            self.update_current_view(False)

    def btn_projection_copy_prev_click(self, button):
        if self.selected_keyframe > 0:
            curr_proj = self.kf_projections[self.selected_keyframe].copy()
            proj = self.kf_projections[self.selected_keyframe - 1].copy()
            self.kf_projections[self.selected_keyframe] = proj
            self.update_projected_cache(self.selected_keyframe)

            to_undo = {
                "operation": "projection_changed",
                "keyframe_idx": self.selected_keyframe,
                "old_projection": curr_proj,
            }
            self.undo_stack.append(to_undo)

            self.update_current_view(False)

    def scroll_delta_x_change(self, scroll):
        self.projection_delta_x = int(scroll.value)
        self.lbl_delta_x.set_text("Delta X: " + str(self.projection_delta_x))
        self.update_projection_image()
        self.update_current_view()

    def scroll_delta_y_change(self, scroll):
        self.projection_delta_y = int(scroll.value)
        self.lbl_delta_y.set_text("Delta Y: " + str(self.projection_delta_y))
        self.update_projection_image()
        self.update_current_view()

    def btn_undo_click(self, button):
        if len(self.undo_stack) == 0:
            print("No operations to undo")
            return

        # copy last operation
        to_undo = self.undo_stack[-1]

        success = False

        if to_undo["operation"] == "projection_changed":
            # revert to previous projection
            affected_keyframe = to_undo["keyframe_idx"]
            old_projection = to_undo["old_projection"]
            # print(old_projection.GenerateXML())
            curr_projection = self.kf_projections[affected_keyframe].copy()
            self.kf_projections[affected_keyframe] = old_projection.copy()
            to_undo["old_projection"] = curr_projection

            self.update_projected_cache(affected_keyframe)
            self.update_current_view(False)

            success = True

        # removing ...
        if success:
            self.redo_stack.append(to_undo)
            del self.undo_stack[-1]

            # update interface ...
            self.update_current_view(False)
        else:
            print("Action could not be undone")

    def btn_redo_click(self, button):
        if len(self.redo_stack) == 0:
            print("No operations to be re-done")
            return

        # copy last operation
        to_redo = self.redo_stack[-1]

        success = False

        if to_redo["operation"] == "projection_changed":
            # revert to previous projection
            affected_keyframe = to_redo["keyframe_idx"]
            new_projection = to_redo["old_projection"].copy()
            curr_projection = self.kf_projections[affected_keyframe].copy()
            self.kf_projections[affected_keyframe] = new_projection
            to_redo["old_projection"] = curr_projection

            self.update_projected_cache(affected_keyframe)
            self.update_current_view(False)

            success = True

        # removing ...
        if success:
            self.undo_stack.append(to_redo)
            del self.redo_stack[-1]

            # update interface ...
            self.update_current_view(False)
        else:
            print("Action could not be re-done")

    def btn_save_click(self, button):
        xml_str = "<ProjectionAnnotations>\n"
        # xml_str += KeyFrameProjection.GenerateKeyFramesProjectionsXML(self.kf_projections)
        # xml_str += SegmentationTree.SegmentationTreesToXML(self.word_trees)
        xml_str += KeyFrameProjection.GenerateKeyFramesProjectionsXML(self.kf_projections)
        xml_str += "</ProjectionAnnotations>\n"

        word_annotations_filename = self.output_path + "/projections.xml"
        out_file = open(word_annotations_filename, "w")
        out_file.write(xml_str)
        out_file.close()

        # clean the action stack
        self.undo_stack = []
        self.redo_stack = []

        print("Saved to: " + word_annotations_filename)

    def btn_exit_click(self, button):
        if len(self.undo_stack) > 0:
            # confirm before losing changes
            self.set_editor_mode(GTProjectionAnnotator.ModeExitConfirm)
        else:
            # Just exit
            self.return_screen = None
            print("APPLICATION FINISHED")
