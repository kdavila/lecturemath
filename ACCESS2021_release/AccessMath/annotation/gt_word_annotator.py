
import os
from copy import deepcopy

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
from AccessMath.annotation.keyframe_words import KeyFrameWords
from AccessMath.preprocessing.content.segmentation_tree import SegmentationTree

class GTWordAnnotator(Screen):
    ModeNavigate = 0
    ModeTreeSelectSplit = 1
    ModeTreeAutoSplit = 2
    ModeTreeEditSplit = 3
    ModeTreeEditMerge = 4
    ModeExitConfirm = 5

    ViewModeNormalRGB = 0
    ViewModeNormalBin = 1
    ViewModeProjectedRGB = 2
    ViewModeProjectedBin = 3

    ParamsMinSplitAlpha = -5.0
    ParamsMaxSplitAlpha = 5.0
    ParamsStepSplitAlpha = 0.05
    ParamsDefSplitAlphaX = -1.25
    ParamsDefSplitAlphaY = -3.00

    def __init__(self, size, db_name, lecture_title, output_path):
        Screen.__init__(self, "Word Ground Truth Annotation Interface", size)

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
        self.keyframe_annotations, self.segments = KeyFrameAnnotation.LoadExportedKeyframes(segments_filename,
                                                                                            keyframes_image_prefix, True)

        if len(self.keyframe_annotations) > 0:
            print("Key-frames Loaded: " + str(len(self.keyframe_annotations)))
        else:
            raise Exception("Cannot start with 0 key-frames")

        self.unprojected_RGB_cache = {}
        self.unprojected_BIN_cache = {}

        # try loading projections ...
        proj_filename = self.output_path + "/projections.xml"
        if os.path.exists(proj_filename):
            print("Loading saved projection annotations")
            self.kf_projections = KeyFrameProjection.LoadKeyFramesProjectionsFromXML(proj_filename, "")

            # apply any projection transformation ... to the raw images
            for idx in range(len(self.keyframe_annotations)):
                # cache original image ...
                self.unprojected_RGB_cache[idx] = self.keyframe_annotations[idx].raw_image

                raw_image, _, obj_mask = self.kf_projections[idx].warpKeyFrame(self.keyframe_annotations[idx], True)
                # update raw image ...
                self.keyframe_annotations[idx].raw_image = raw_image
                # update gray scale ....
                self.keyframe_annotations[idx].update_grayscale()
                # update object mask ....
                self.keyframe_annotations[idx].object_mask = obj_mask
                # update combined image ...
                self.keyframe_annotations[idx].update_combined_image()
        else:
            raise Exception("Cannot start without Projection Annotations")

        portions_filename = self.output_path + "/portions.xml"
        portions_path = self.output_path + "/portions/"
        if os.path.exists(portions_filename):
            # Saved data detected, loading
            print("Previously saved portion data detected, loading")
            KeyFrameAnnotation.LoadKeyframesPortions(portions_filename, self.keyframe_annotations, portions_path)
        else:
            raise Exception("No saved portion data detected, cannot continue")

        # key-frames will not be combined in this mode
        print("Original Key-frames: " + str(len(self.keyframe_annotations)))
        print("Segments: " + str(len(self.segments)))

        # getting the un-projected binary ....
        for idx in range(len(self.keyframe_annotations)):
            raw_bin = self.kf_projections[idx].warpImage(self.keyframe_annotations[idx].binary_image,True)
            self.unprojected_BIN_cache[idx] = raw_bin

        # Loading data specific to the Word Annotation process ...
        self.kf_words = []

        # .... try loading from file ....
        words_filename = self.output_path + "/word_annotations.xml"

        if os.path.exists(words_filename):
            print("Loading saved word-level annotations")

            binary_images = [255 - kf.binary_image[:, :, 0] for kf in self.keyframe_annotations]
            all_words = SegmentationTree.LoadSegmentationTreesFromXML(words_filename,"", binary_images)
            for idx in range(len(self.keyframe_annotations)):
                words = KeyFrameWords(self.keyframe_annotations[idx], self.kf_projections[idx], all_words[idx])
                self.kf_words.append(words)

        else:
            print("No previous Word-level annotations found")
            # ... no data found? create defaults ...
            for idx in range(len(self.keyframe_annotations)):

                inv_binary = 255 - self.keyframe_annotations[idx].binary_image
                def_segment = SegmentationTree.CreateDefault(inv_binary)

                def_words = KeyFrameWords(self.keyframe_annotations[idx], self.kf_projections[idx], def_segment)
                self.kf_words.append(def_words)

        # Creating interface ...
        self.view_mode = GTWordAnnotator.ViewModeNormalRGB
        self.edition_mode = GTWordAnnotator.ModeNavigate
        self.view_scale = 1.0
        self.selected_keyframe = 0

        self.tempo_word_tree = None
        self.tempo_word_tree_node = None

        self.auto_split_alpha_x = -1.25
        self.auto_split_alpha_y = -3.00

        self.manual_split_vertical = False

        # add elements....
        container_top = 10
        container_width = 330

        button_2_width = 150
        button_2_left = int(container_width * 0.25) - button_2_width / 2
        button_2_right = int(container_width * 0.75) - button_2_width / 2

        # ======================================================================
        #   Navigation panel to move across frames
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

        self.btn_view_normal_bin = ScreenButton("btn_view_normal_bin", "Normal BIN", 21, button_2_width)
        self.btn_view_normal_bin.set_colors(button_text_color, button_back_color)
        self.btn_view_normal_bin.position = (button_2_right, self.btn_zoom_zero.get_bottom() + 10)
        self.btn_view_normal_bin.click_callback = self.btn_view_normal_bin_click
        self.container_view_buttons.append(self.btn_view_normal_bin)

        self.btn_view_projected_rgb = ScreenButton("btn_view_projected_rgb", "Projected RGB", 21, button_2_width)
        self.btn_view_projected_rgb.set_colors(button_text_color, button_back_color)
        self.btn_view_projected_rgb.position = (button_2_left, self.btn_view_normal_bin.get_bottom() + 10)
        self.btn_view_projected_rgb.click_callback = self.btn_view_projected_rgb_click
        self.container_view_buttons.append(self.btn_view_projected_rgb)

        self.btn_view_projected_bin = ScreenButton("btn_view_projected_bin", "Projected BIN", 21, button_2_width)
        self.btn_view_projected_bin.set_colors(button_text_color, button_back_color)
        self.btn_view_projected_bin.position = (button_2_right, self.btn_view_normal_bin.get_bottom() + 10)
        self.btn_view_projected_bin.click_callback = self.btn_view_projected_bin_click
        self.container_view_buttons.append(self.btn_view_projected_bin)

        # ==============================================

        self.container_auto_split = ScreenContainer("container_auto_split", (container_width, 150), general_background)
        self.container_auto_split.position = (self.container_view_buttons.get_left(),
                                              self.container_view_buttons.get_bottom() + 5)
        self.elements.append(self.container_auto_split)

        self.lbl_auto_split_title = ScreenLabel("lbl_auto_split_title", "Auto-Split Parameters", 21,
                                                container_width - 10, 1)
        self.lbl_auto_split_title.position = (5, 5)
        self.lbl_auto_split_title.set_background(general_background)
        self.lbl_auto_split_title.set_color(text_color)
        self.container_auto_split.append(self.lbl_auto_split_title)

        self.lbl_alpha_x = ScreenLabel("lbl_alpha_x", "Alpha X: {0:.2f}".format(self.auto_split_alpha_x), 21,
                                       container_width - 10, 1)
        self.lbl_alpha_x.position = (5, self.lbl_auto_split_title.get_bottom() + 20)
        self.lbl_alpha_x.set_background(general_background)
        self.lbl_alpha_x.set_color(text_color)
        self.container_auto_split.append(self.lbl_alpha_x)

        min_alpha = GTWordAnnotator.ParamsMinSplitAlpha
        max_alpha = GTWordAnnotator.ParamsMaxSplitAlpha
        step_alpha = GTWordAnnotator.ParamsStepSplitAlpha

        def_alpha_x = GTWordAnnotator.ParamsDefSplitAlphaX
        self.scroll_alpha_x = ScreenHorizontalScroll("scroll_alpha_x", min_alpha, max_alpha, def_alpha_x, step_alpha)
        self.scroll_alpha_x.position = (5, self.lbl_alpha_x.get_bottom() + 10)
        self.scroll_alpha_x.width = container_width - 10
        self.scroll_alpha_x.scroll_callback = self.scroll_alpha_x_change
        self.container_auto_split.append(self.scroll_alpha_x)

        self.lbl_alpha_y = ScreenLabel("lbl_alpha_y", "Alpha Y: {0:.2f}".format(self.auto_split_alpha_y), 21,
                                       container_width - 10, 1)
        self.lbl_alpha_y.position = (5, self.scroll_alpha_x.get_bottom() + 20)
        self.lbl_alpha_y.set_background(general_background)
        self.lbl_alpha_y.set_color(text_color)
        self.container_auto_split.append(self.lbl_alpha_y)

        def_alpha_y = GTWordAnnotator.ParamsDefSplitAlphaY
        self.scroll_alpha_y = ScreenHorizontalScroll("scroll_alpha_y", min_alpha, max_alpha, def_alpha_y, step_alpha)
        self.scroll_alpha_y.position = (5, self.lbl_alpha_y.get_bottom() + 10)
        self.scroll_alpha_y.width = container_width - 10
        self.scroll_alpha_y.scroll_callback = self.scroll_alpha_y_change
        self.container_auto_split.append(self.scroll_alpha_y)

        self.container_auto_split.visible = False

        # ===============================================
        # Panel for word segmentation buttons
        self.container_segmentation_buttons = ScreenContainer("container_segmentation_buttons", (container_width, 230),
                                                            darker_background)
        self.container_segmentation_buttons.position = (self.container_view_buttons.get_left(),
                                                        self.container_view_buttons.get_bottom() + 5)
        self.elements.append(self.container_segmentation_buttons)

        self.btn_segmentation_auto = ScreenButton("btn_segmentation_auto", "Auto Segmentation", 21, button_width)
        self.btn_segmentation_auto.set_colors(button_text_color, button_back_color)
        self.btn_segmentation_auto.position = (button_left, 5)
        self.btn_segmentation_auto.click_callback = self.btn_segmentation_auto_click
        self.container_segmentation_buttons.append(self.btn_segmentation_auto)

        self.btn_segmentation_split_hor = ScreenButton("btn_segmentation_split_hor", "Hor. Split", 21, button_2_width)
        self.btn_segmentation_split_hor.set_colors(button_text_color, button_back_color)
        self.btn_segmentation_split_hor.position = (button_2_left, self.btn_segmentation_auto.get_bottom() + 10)
        self.btn_segmentation_split_hor.click_callback = self.btn_segmentation_split_hor_click
        self.container_segmentation_buttons.append(self.btn_segmentation_split_hor)

        self.btn_segmentation_split_ver = ScreenButton("btn_segmentation_split_ver", "Ver. Split", 21, button_2_width)
        self.btn_segmentation_split_ver.set_colors(button_text_color, button_back_color)
        self.btn_segmentation_split_ver.position = (button_2_right, self.btn_segmentation_auto.get_bottom() + 10)
        self.btn_segmentation_split_ver.click_callback = self.btn_segmentation_split_ver_click
        self.container_segmentation_buttons.append(self.btn_segmentation_split_ver)

        self.btn_segmentation_copy_prev = ScreenButton("btn_segmentation_copy_prev", "Copy Prev.", 21, button_2_width)
        self.btn_segmentation_copy_prev.set_colors(button_text_color, button_back_color)
        self.btn_segmentation_copy_prev.position = (button_2_left, self.btn_segmentation_split_hor.get_bottom() + 10)
        self.btn_segmentation_copy_prev.click_callback = self.btn_segmentation_copy_prev_click
        self.container_segmentation_buttons.append(self.btn_segmentation_copy_prev)

        self.btn_segmentation_copy_next = ScreenButton("btn_segmentation_copy_next", "Copy Next.", 21, button_2_width)
        self.btn_segmentation_copy_next.set_colors(button_text_color, button_back_color)
        self.btn_segmentation_copy_next.position = (button_2_right, self.btn_segmentation_split_hor.get_bottom() + 10)
        self.btn_segmentation_copy_next.click_callback = self.btn_segmentation_copy_next_click
        self.container_segmentation_buttons.append(self.btn_segmentation_copy_next)

        self.btn_segmentation_merge = ScreenButton("btn_segmentation_merge", "Merge Segments", 21, button_width)
        self.btn_segmentation_merge.set_colors(button_text_color, button_back_color)
        self.btn_segmentation_merge.position = (button_left, self.btn_segmentation_copy_prev.get_bottom() + 10)
        self.btn_segmentation_merge.click_callback = self.btn_segmentation_merge_click
        self.container_segmentation_buttons.append(self.btn_segmentation_merge)

        self.btn_segmentation_reset = ScreenButton("btn_segmentation_reset", "Reset Segments", 21, button_width)
        self.btn_segmentation_reset.set_colors(button_text_color, button_back_color)
        self.btn_segmentation_reset.position = (button_left, self.btn_segmentation_merge.get_bottom() + 10)
        self.btn_segmentation_reset.click_callback = self.btn_segmentation_reset_click
        self.container_segmentation_buttons.append(self.btn_segmentation_reset)

        #=============================================================
        # Panel with state buttons (Undo, Redo, Save, Export)
        self.container_state_buttons = ScreenContainer("container_state_buttons", (container_width, 250),
                                                       general_background)
        self.container_state_buttons.position = (
        self.container_view_buttons.get_left(), self.container_segmentation_buttons.get_bottom() + 10)
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
        self.img_main.double_click_callback = self.img_mouse_double_click
        self.container_images.append(self.img_main)

        # canvas used for annotations
        self.canvas_select = ScreenCanvas("canvas_select", 100, 100)
        self.canvas_select.position = (0, 0)
        self.canvas_select.locked = True
        # self.canvas_select.object_edited_callback = self.canvas_object_edited
        # self.canvas_select.object_selected_callback = self.canvas_selection_changed
        self.container_images.append(self.canvas_select)

        self.canvas_select.add_polygon_element("selection_polygon", self.kf_words[0].projection.base_dst_points)
        self.canvas_select.elements["selection_polygon"].visible = False

        self.undo_stack = []
        self.redo_stack = []

        self.elements.key_up_callback = self.main_key_up

        self.update_current_view(True)

    def update_current_view(self, resized=False):
        if self.view_mode == GTWordAnnotator.ViewModeNormalBin:
            # get un-projected binary .. .
            base_image = self.unprojected_BIN_cache[self.selected_keyframe]
        elif self.view_mode == GTWordAnnotator.ViewModeProjectedRGB:
            # the key-frame stores RGB in projection space ...
            base_image = self.keyframe_annotations[self.selected_keyframe].raw_image
        elif self.view_mode == GTWordAnnotator.ViewModeProjectedBin:
            # the key-frame stores binary in projection space ...
            base_image = self.keyframe_annotations[self.selected_keyframe].binary_image
        else:
            # default for ViewModeNormalRGB ... from original RGB
            base_image = self.unprojected_RGB_cache[self.selected_keyframe]

        h, w, c = base_image.shape
        modified_image = base_image.copy()

        # display the boxes as ....
        # get the boxes and convert to polygon
        if (self.edition_mode == GTWordAnnotator.ModeTreeSelectSplit or
            self.edition_mode == GTWordAnnotator.ModeTreeAutoSplit or
            self.edition_mode == GTWordAnnotator.ModeTreeEditMerge or
            self.edition_mode == GTWordAnnotator.ModeTreeEditSplit):
            # dynamically editing the current segmentation tree
            current_bboxes = self.tempo_word_tree.collect_all_leaves()
        else:
            # TODO: cache this step?
            current_bboxes = self.kf_words[self.selected_keyframe].get_words()

        projection = self.kf_words[self.selected_keyframe].projection
        polygons = projection.bboxesToPolygons(current_bboxes)

        if (self.view_mode == GTWordAnnotator.ViewModeProjectedRGB or
            self.view_mode == GTWordAnnotator.ViewModeProjectedBin):
            # use the original bboxes ... no projections needed in this space
            current_polygons = polygons
        else:
            current_polygons = projection.warpPolygons(polygons, True)

        mod_polygons = [polygon.reshape((-1, 1, 2)).astype(np.int32) for polygon in current_polygons]
        cv2.polylines(modified_image, mod_polygons, True, (0, 255, 0), thickness=2)

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

    def btn_confirm_cancel_click(self, button):
        # by default, got back to navigation mode ...
        self.set_editor_mode(GTWordAnnotator.ModeNavigate)
        self.update_current_view(False)

    def commit_current_tree_changes(self):
        old_segment = deepcopy(self.kf_words[self.selected_keyframe].segment_tree)
        # accept change .... replace with temporal word tree ...
        self.kf_words[self.selected_keyframe].segment_tree = self.tempo_word_tree

        to_undo = {
            "operation": "tree_changed",
            "keyframe_idx": self.selected_keyframe,
            "old_segmentation": old_segment,
        }
        self.undo_stack.append(to_undo)
        self.redo_stack = []

        # update the view ....
        self.set_editor_mode(GTWordAnnotator.ModeNavigate)
        self.update_current_view(False)

    def btn_confirm_accept_click(self, button):
        if (self.edition_mode == GTWordAnnotator.ModeTreeAutoSplit or
              self.edition_mode == GTWordAnnotator.ModeTreeEditSplit or
              self.edition_mode == GTWordAnnotator.ModeTreeEditMerge):

            self.commit_current_tree_changes()

        elif self.edition_mode == GTWordAnnotator.ModeExitConfirm:
            # exit
            self.return_screen = None
            print("Changes have been lost!")
            print("APPLICATION FINISHED")

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
        self.view_mode = GTWordAnnotator.ViewModeNormalRGB
        self.update_current_view()

    def btn_view_normal_bin_click(self, button):
        self.view_mode = GTWordAnnotator.ViewModeNormalBin
        self.update_current_view()

    def btn_view_projected_rgb_click(self, button):
        self.view_mode = GTWordAnnotator.ViewModeProjectedRGB
        self.update_current_view()

    def btn_view_projected_bin_click(self, button):
        self.view_mode = GTWordAnnotator.ViewModeProjectedBin
        self.update_current_view()

    def btn_segmentation_auto_click(self, button):
        # current_tree = self.word_trees[self.selected_keyframe]
        # current_tree.segment(current_tree.root)
        self.tempo_word_tree = deepcopy(self.kf_words[self.selected_keyframe].segment_tree)
        self.tempo_word_tree_node = None
        self.set_editor_mode(GTWordAnnotator.ModeTreeSelectSplit)
        self.update_current_view(False)

    def btn_segmentation_split_hor_click(self, button):
        self.tempo_word_tree = deepcopy(self.kf_words[self.selected_keyframe].segment_tree)
        self.tempo_word_tree_node = None
        self.manual_split_vertical = False
        self.set_editor_mode(GTWordAnnotator.ModeTreeEditSplit)
        self.update_current_view(False)

    def btn_segmentation_split_ver_click(self, button):
        self.tempo_word_tree = deepcopy(self.kf_words[self.selected_keyframe].segment_tree)
        self.tempo_word_tree_node = None
        self.manual_split_vertical = True
        self.set_editor_mode(GTWordAnnotator.ModeTreeEditSplit)
        self.update_current_view(False)

    def btn_segmentation_merge_click(self, button):
        self.tempo_word_tree = deepcopy(self.kf_words[self.selected_keyframe].segment_tree)
        self.tempo_word_tree_node = None
        self.set_editor_mode(GTWordAnnotator.ModeTreeEditMerge)
        self.update_current_view(False)

    def btn_undo_click(self, button):
        if len(self.undo_stack) == 0:
            print("No operations to undo")
            return

        # copy last operation
        to_undo = self.undo_stack[-1]

        success = False

        if to_undo["operation"] == "tree_changed":
            # revert to previous segmentation
            affected_keyframe = to_undo["keyframe_idx"]

            old_segment = to_undo["old_segmentation"]
            curr_segment = deepcopy(self.kf_words[self.selected_keyframe].segment_tree)

            self.kf_words[affected_keyframe].segment_tree = deepcopy(old_segment)
            to_undo["old_segmentation"] = curr_segment

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

        if to_redo["operation"] == "tree_changed":
            # revert to previous segmentation
            affected_keyframe = to_redo["keyframe_idx"]

            new_segmentation = deepcopy(to_redo["old_segmentation"])
            curr_segmentation = deepcopy(self.kf_words[affected_keyframe].segment_tree)

            self.kf_words[affected_keyframe].segment_tree = new_segmentation
            to_redo["old_segmentation"] = curr_segmentation

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

        word_trees = [kf_word.segment_tree for kf_word in self.kf_words]

        xml_str = "<WordAnnotations>\n"
        # xml_str += KeyFrameProjection.GenerateKeyFramesProjectionsXML(self.kf_projections)
        xml_str += SegmentationTree.SegmentationTreesToXML(word_trees)
        # xml_str += KeyFrameWords.KeyFramesWordsToXML(self.kf_words)
        xml_str += "</WordAnnotations>\n"

        word_annotations_filename = self.output_path + "/word_annotations.xml"
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
            self.set_editor_mode(GTWordAnnotator.ModeExitConfirm)
        else:
            # Just exit
            self.return_screen = None
            print("APPLICATION FINISHED")

    def set_editor_mode(self, new_mode):
        self.edition_mode = new_mode
        self.container_nav_buttons.visible = (new_mode == GTWordAnnotator.ModeNavigate)

        self.container_confirm_buttons.visible = (new_mode == GTWordAnnotator.ModeTreeSelectSplit or
                                                  new_mode == GTWordAnnotator.ModeTreeAutoSplit or
                                                  new_mode == GTWordAnnotator.ModeTreeEditMerge or
                                                  new_mode == GTWordAnnotator.ModeTreeEditSplit or
                                                  new_mode == GTWordAnnotator.ModeExitConfirm)

        if new_mode == GTWordAnnotator.ModeTreeSelectSplit:
            self.lbl_confirm_message.set_text("Select Node to Auto-split")
        elif new_mode == GTWordAnnotator.ModeTreeAutoSplit:
            self.lbl_confirm_message.set_text("Auto-Split Node")
        elif new_mode == GTWordAnnotator.ModeTreeEditSplit:
            if self.manual_split_vertical:
                self.lbl_confirm_message.set_text("Split Words (Cut X Coord)")
            else:
                self.lbl_confirm_message.set_text("Split Words (Cut Y Coord)")
        elif new_mode == GTWordAnnotator.ModeTreeEditMerge:
            self.lbl_confirm_message.set_text("Merge Words")
        elif new_mode == GTWordAnnotator.ModeExitConfirm:
            self.lbl_confirm_message.set_text("Exit Without Saving?")

        # for every other mode
        self.canvas_select.locked = True
        self.canvas_select.elements["selection_polygon"].visible = False

        self.container_state_buttons.visible = (new_mode == GTWordAnnotator.ModeNavigate)

        self.container_segmentation_buttons.visible = (new_mode == GTWordAnnotator.ModeNavigate)

        self.container_auto_split.visible = (new_mode == GTWordAnnotator.ModeTreeAutoSplit)

    def scroll_alpha_x_change(self, scroll):
        self.lbl_alpha_x.set_text("Alpha X: {0:.2f}".format(scroll.value))
        self.auto_split_tempo_node()
        self.update_current_view()

    def scroll_alpha_y_change(self, scroll):
        self.lbl_alpha_y.set_text("Alpha Y: {0:.2f}".format(scroll.value))
        self.auto_split_tempo_node()
        self.update_current_view()

    def node_from_click(self, click_x, click_y):
        # first, project the click from screen space to segment space (if needed)
        if self.view_mode == GTWordAnnotator.ViewModeNormalBin or self.view_mode == GTWordAnnotator.ViewModeNormalRGB:
            warp_x, warp_y = self.kf_words[self.selected_keyframe].projection.warpPoint(click_x, click_y, False)
        else:
            warp_x, warp_y = click_x, click_y

        # print((warp_x, warp_y))
        # current_tree = self.word_trees[self.selected_keyframe]
        bbox, node = self.tempo_word_tree.find_bbox_by_coords(warp_x, warp_y, self.tempo_word_tree.root, True)

        # print(((warp_x, warp_y), bbox))
        return node, (warp_x, warp_y)

    def auto_split_tempo_node(self):
        # if the selected node is not a leaf... convert into a leaf
        if not self.tempo_word_tree_node.is_leaf:
            # it has children ... remove them (by calling removal of one child)
            self.tempo_word_tree.remove_segment(self.tempo_word_tree_node.left)

        # call segment at the current node again ... with the current alpha values
        # print((self.scroll_alpha_x.value, self.scroll_alpha_y.value))
        self.tempo_word_tree.segment(self.tempo_word_tree_node, self.scroll_alpha_x.value, self.scroll_alpha_y.value)

    def img_mouse_down(self, img_object, pos, button):
        if button == 1:
            # ... first, get click location on original image space
            scaled_x, scaled_y = pos
            click_x = int(scaled_x / self.view_scale)
            click_y = int(scaled_y / self.view_scale)

            if self.edition_mode == GTWordAnnotator.ModeTreeSelectSplit:
                # print((click_x, click_y))

                selected_node, _ = self.node_from_click(click_x, click_y)
                if selected_node is not None:
                    self.tempo_word_tree_node = selected_node
                    self.auto_split_tempo_node()

                    self.set_editor_mode(GTWordAnnotator.ModeTreeAutoSplit)
                    self.update_current_view()

            elif self.edition_mode == GTWordAnnotator.ModeTreeEditSplit:

                selected_node, (w_x, w_y) = self.node_from_click(click_x, click_y)
                if selected_node is not None:
                    if self.manual_split_vertical:
                        # make a vertical cut (at X value)
                        self.tempo_word_tree.force_segment_X(w_x, selected_node)
                    else:
                        # make a horizontal cut (at Y value)
                        self.tempo_word_tree.force_segment_Y(w_y, selected_node)

                    self.update_current_view()

            elif self.edition_mode == GTWordAnnotator.ModeTreeEditMerge:

                selected_node, (w_x, w_y) = self.node_from_click(click_x, click_y)
                if selected_node is not None and selected_node.parent is not None:
                    # something was selected and is not the root ...
                    self.tempo_word_tree.remove_segment(selected_node)

                    self.update_current_view()

    def btn_segmentation_copy_prev_click(self, button):
        if self.selected_keyframe > 0:
            # get copy on temporal tree ....
            self.tempo_word_tree = deepcopy(self.kf_words[self.selected_keyframe - 1].segment_tree)
            # update image references and derivated data ...
            inv_binary = 255 - self.keyframe_annotations[self.selected_keyframe].binary_image[:,:, 0]
            self.tempo_word_tree.update_image(inv_binary)
            # self.tempo_word_tree
            self.commit_current_tree_changes()

    def btn_segmentation_copy_next_click(self, button):
        if self.selected_keyframe + 1 < len(self.kf_words):
            # get copy on temporal tree ....
            self.tempo_word_tree = deepcopy(self.kf_words[self.selected_keyframe + 1].segment_tree)
            # update image references and derivated data ...
            inv_binary = 255 - self.keyframe_annotations[self.selected_keyframe].binary_image[:,:, 0]
            print(inv_binary.shape)
            self.tempo_word_tree.update_image(inv_binary)
            # self.tempo_word_tree
            self.commit_current_tree_changes()

    def btn_segmentation_reset_click(self, button):
        # reseting the tree ...
        inv_binary = 255 - self.keyframe_annotations[self.selected_keyframe].binary_image
        self.tempo_word_tree = SegmentationTree.CreateDefault(inv_binary)
        self.commit_current_tree_changes()

    def main_key_up(self, scancode, key, unicode):
        # key short cuts
        if key == 269:
            # minus
            if self.container_view_buttons.visible:
                self.btn_zoom_reduce_click(None)
        elif key == 270:
            # plus
            if self.container_view_buttons.visible:
                self.btn_zoom_increase_click(None)
        elif key == 110:
            # n
            if self.container_nav_buttons.visible:
                self.btn_nav_keyframe_next_click(None)
        elif key == 112:
            # p
            if self.container_nav_buttons.visible:
                self.btn_nav_keyframe_prev_click(None)
        elif key == 276:
            # Left key
            pass
        elif key == 275:
            # Right key
            pass

        elif key == 13 or key == 271:
            # return key
            if self.container_confirm_buttons.visible:
                self.btn_confirm_accept_click(None)
        elif key == 27:
            # escape key ..
            if self.container_confirm_buttons.visible:
                self.btn_confirm_cancel_click(None)
        else:
            # print(key)
            pass

    def img_mouse_double_click(self, element, pos, button):
        if button == 1:
            # double left click ...
            if self.edition_mode == GTWordAnnotator.ModeNavigate:
                scaled_x, scaled_y = pos
                click_x = int(scaled_x / self.view_scale)
                click_y = int(scaled_y / self.view_scale)

                selected_node, _ = self.node_from_click(click_x, click_y)
                if selected_node is not None:
                    self.tempo_word_tree_node = selected_node
                    self.auto_split_tempo_node()

                    self.set_editor_mode(GTWordAnnotator.ModeTreeAutoSplit)
                    self.update_current_view()

