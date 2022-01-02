
import os
import math

import cv2
import numpy as np
from scipy import interpolate

import xml.etree.ElementTree as ET

from AM_CommonTools.util.time_helper import TimeHelper
from AM_CommonTools.interface.controls.screen import Screen
from AM_CommonTools.interface.controls.screen_button import ScreenButton
from AM_CommonTools.interface.controls.screen_canvas import ScreenCanvas
from AM_CommonTools.interface.controls.screen_container import ScreenContainer
from AM_CommonTools.interface.controls.screen_horizontal_scroll import ScreenHorizontalScroll
from AM_CommonTools.interface.controls.screen_image import ScreenImage
from AM_CommonTools.interface.controls.screen_label import ScreenLabel

from AccessMath.annotation.video_object import VideoObject
from AccessMath.annotation.keyframe_annotation import KeyFrameAnnotation
from AccessMath.annotation.keyframe_projection import KeyFrameProjection
from AccessMath.annotation.keyframe_words import KeyFrameWords
from AccessMath.annotation.drawing_info import DrawingInfo
from AccessMath.annotation.unique_word_group import UniqueWordGroup
from AccessMath.preprocessing.content.segmentation_tree import SegmentationTree
from AccessMath.util.visualizer import Visualizer

class GTUniqueWordAnnotator(Screen):
    ModeNavigate = 0
    ModeMatch_RegionSelection = 1
    ModeMatch_Matching = 2
    ModeMatch_Remove = 3
    ModeExitConfirm = 4

    ViewModeNormalRGB = 0
    ViewModeNormalBin = 1
    ViewModeProjectedRGB = 2
    ViewModeProjectedBin = 3

    WordsShowColors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                       (255, 255, 0), (255, 0, 255), (0, 255, 255),
                       (128, 0, 0), (0, 128, 0), (0, 0, 128),
                       (128, 128, 0), (128, 0, 128), (0, 128, 128),
                       (255, 128, 0), (255, 0, 128), (0, 255, 128),
                       (128, 255, 0), (128, 0, 255), (0, 128, 255)]

    ParamsMinIOU = 10
    ParamsMaxTranslation = 10

    def __init__(self, size, db_name, lecture_title, output_path):
        Screen.__init__(self, "Unique Words Ground Truth Annotation Interface", size)

        general_background = (100, 90, 80)
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
        self.keyframe_annotations, self.segments = KeyFrameAnnotation.LoadExportedKeyframes(segments_filename, keyframes_image_prefix, True)

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

        print("Original Key-frames: " + str(len(self.keyframe_annotations)))
        print("Segments: " + str(len(self.segments)))

        # getting the un-projected binary ....
        for idx in range(len(self.keyframe_annotations)):
            raw_bin = self.kf_projections[idx].warpImage(self.keyframe_annotations[idx].binary_image, True)
            self.unprojected_BIN_cache[idx] = raw_bin

        # Load projections and segmentation trees  ...
        self.kf_words = []

        # .... try loading from file ....
        words_filename = self.output_path + "/word_annotations.xml"

        if os.path.exists(words_filename):
            print("Loading saved word-level annotations")
            binary_images = [255 - kf.binary_image[:, :, 0] for kf in self.keyframe_annotations]
            all_words = SegmentationTree.LoadSegmentationTreesFromXML(words_filename, "", binary_images)
            for idx in range(len(self.keyframe_annotations)):
                words = KeyFrameWords(self.keyframe_annotations[idx], self.kf_projections[idx], all_words[idx])
                self.kf_words.append(words)
        else:
            raise Exception("No previous Word-level annotations found")

        # other Word/group elements
        self.unique_groups = None
        self.word_group = None
        self.word_total = 0
        self.collected_words = []
        for kf_idx, keyframe in enumerate(self.keyframe_annotations):
            current_words = self.kf_words[kf_idx].get_words()
            self.collected_words.append(current_words)
            self.word_total += len(current_words)

        unique_words_filename = self.output_path + "/unique_words.xml"

        if os.path.exists(unique_words_filename):
            # Saved data detected, loading
            print("Previously saved unique words data detected, loading")

            self.word_group, self.unique_groups = UniqueWordGroup.GroupsFromXML(self.kf_words, unique_words_filename)
        else:
            # no previous data, build default index (all CCs are unique)
            self.unique_groups = []
            self.word_group = []
            for kf_idx, keyframe in enumerate(self.kf_words):
                self.word_group.append({})

                for word in self.collected_words[kf_idx]:
                    new_group = UniqueWordGroup(word, kf_idx)
                    self.unique_groups.append(new_group)
                    self.word_group[kf_idx][UniqueWordGroup.wordID(word)] = new_group

        self.view_mode = GTUniqueWordAnnotator.ViewModeNormalRGB
        self.edition_mode = GTUniqueWordAnnotator.ModeNavigate
        self.view_scale = 1.0
        self.selected_keyframe = 0

        self.matching_delta_x = 0
        self.matching_delta_y = 0
        self.matching_scores = None
        self.matching_min_IOU = 0.99
        self.base_matching = None

        # add elements....
        container_top = 10
        container_width = 330

        button_2_width = 150
        button_2_left = int(container_width * 0.25) - button_2_width / 2
        button_2_right = int(container_width * 0.75) - button_2_width / 2

        # ========================================================
        # Navigation panel to move accross frames
        self.container_nav_buttons = ScreenContainer("container_nav_buttons", (container_width, 70), back_color=general_background)
        self.container_nav_buttons.position = (self.width - self.container_nav_buttons.width - 10, container_top)
        self.elements.append(self.container_nav_buttons)

        self.lbl_nav_keyframe = ScreenLabel("lbl_nav_keyframe", "Key-Frame: 1 / " + str(len(self.keyframe_annotations)), 21, 290, 1)
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

        # ========================================================
        # confirmation panel
        self.container_confirm_buttons = ScreenContainer("container_confirm_buttons", (container_width, 70), back_color=general_background)
        self.container_confirm_buttons.position = (self.width - self.container_confirm_buttons.width - 10, container_top)
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

        # ========================================================
        # View panel with view control buttons
        self.container_view_buttons = ScreenContainer("container_view_buttons", (container_width, 165), back_color=general_background)
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

        # ========================================================
        # Panel with action buttons (Add/Remove links)
        self.container_action_buttons = ScreenContainer("container_action_buttons", (container_width, 45),
                                                        general_background)
        self.container_action_buttons.position = (self.container_view_buttons.get_left(), self.container_view_buttons.get_bottom() + 5)
        self.elements.append(self.container_action_buttons)

        self.btn_matches_add = ScreenButton("btn_matches_add", "Add Matches", 21, button_2_width)
        self.btn_matches_add.set_colors(button_text_color, button_back_color)
        self.btn_matches_add.position = (button_2_left, 5)
        self.btn_matches_add.click_callback = self.btn_matches_add_click
        self.container_action_buttons.append(self.btn_matches_add)

        self.btn_matches_del = ScreenButton("btn_matches_del", "Del. Matches", 21, button_2_width)
        self.btn_matches_del.set_colors(button_text_color, button_back_color)
        self.btn_matches_del.position = (button_2_right, 5)
        self.btn_matches_del.click_callback = self.btn_matches_del_click
        self.container_action_buttons.append(self.btn_matches_del)

        # ===============================================
        # Panel with matching parameters for step 1 (Matching Translation)
        self.container_matching_translation = ScreenContainer("container_matching_translation", (container_width, 150), general_background)
        self.container_matching_translation.position = (self.container_view_buttons.get_left(), self.container_view_buttons.get_bottom() + 5)
        self.elements.append(self.container_matching_translation)

        self.lbl_translation_title = ScreenLabel("lbl_translation_title", "Translation Parameters", 21, container_width - 10, 1)
        self.lbl_translation_title.position = (5, 5)
        self.lbl_translation_title.set_background(general_background)
        self.lbl_translation_title.set_color(text_color)
        self.container_matching_translation.append(self.lbl_translation_title)

        self.lbl_delta_x = ScreenLabel("lbl_delta_x", "Delta X: " + str(self.matching_delta_x), 21, container_width - 10, 1)
        self.lbl_delta_x.position = (5, self.lbl_translation_title.get_bottom() + 20)
        self.lbl_delta_x.set_background(general_background)
        self.lbl_delta_x.set_color(text_color)
        self.container_matching_translation.append(self.lbl_delta_x)

        max_delta = GTUniqueWordAnnotator.ParamsMaxTranslation
        self.scroll_delta_x = ScreenHorizontalScroll("scroll_delta_x", -max_delta, max_delta, 0, 1)
        self.scroll_delta_x.position = (5, self.lbl_delta_x.get_bottom() + 10)
        self.scroll_delta_x.width = container_width - 10
        self.scroll_delta_x.scroll_callback = self.scroll_delta_x_change
        self.container_matching_translation.append(self.scroll_delta_x)

        self.lbl_delta_y = ScreenLabel("lbl_delta_y", "Delta Y: " + str(self.matching_delta_y), 21, container_width - 10, 1)
        self.lbl_delta_y.position = (5, self.scroll_delta_x.get_bottom() + 20)
        self.lbl_delta_y.set_background(general_background)
        self.lbl_delta_y.set_color(text_color)
        self.container_matching_translation.append(self.lbl_delta_y)

        self.scroll_delta_y = ScreenHorizontalScroll("scroll_delta_y", -max_delta, max_delta, 0, 1)
        self.scroll_delta_y.position = (5, self.lbl_delta_y.get_bottom() + 10)
        self.scroll_delta_y.width = container_width - 10
        self.scroll_delta_y.scroll_callback = self.scroll_delta_y_change
        self.container_matching_translation.append(self.scroll_delta_y)

        self.container_matching_translation.visible = False

        # ===============================================
        # Panel with matching parameters for step 2 (Matching Strictness)
        self.container_matching_strictness = ScreenContainer("container_matching_strictness", (container_width, 150),
                                                              general_background)
        self.container_matching_strictness.position = (self.container_view_buttons.get_left(), self.container_view_buttons.get_bottom() + 5)
        self.elements.append(self.container_matching_strictness)

        self.lbl_matching_title = ScreenLabel("lbl_matching_title", "Matching Parameters", 21,
                                                 container_width - 10, 1)
        self.lbl_matching_title.position = (5, 5)
        self.lbl_matching_title.set_background(general_background)
        self.lbl_matching_title.set_color(text_color)
        self.container_matching_strictness.append(self.lbl_matching_title)

        str_IOU = "Minimum IOU: " + str(int(self.matching_min_IOU * 100))
        self.lbl_min_IOU = ScreenLabel("lbl_min_IOU", str_IOU, 21, container_width - 10, 1)
        self.lbl_min_IOU.position = (5, self.lbl_matching_title.get_bottom() + 20)
        self.lbl_min_IOU.set_background(general_background)
        self.lbl_min_IOU.set_color(text_color)
        self.container_matching_strictness.append(self.lbl_min_IOU)

        min_IOU = GTUniqueWordAnnotator.ParamsMinIOU
        self.scroll_min_IOU = ScreenHorizontalScroll("scroll_min_IOU", min_IOU, 100, 99, 1)
        self.scroll_min_IOU.position = (5, self.lbl_min_IOU.get_bottom() + 10)
        self.scroll_min_IOU.width = container_width - 10
        self.scroll_min_IOU.scroll_callback = self.scroll_min_IOU_change
        self.container_matching_strictness.append(self.scroll_min_IOU)

        self.container_matching_strictness.visible = False

        # ===============================================
        stats_background = (60, 50, 40)
        self.container_stats = ScreenContainer("container_stats", (container_width, 70), back_color=stats_background)
        self.container_stats.position = (self.width - container_width - 10, self.container_action_buttons.get_bottom() + 5)
        self.elements.append(self.container_stats)

        self.lbl_word_stats = ScreenLabel("lbl_cc_stats", "Words Stats", 21, container_width - 10, 1)
        self.lbl_word_stats.position = (5, 5)
        self.lbl_word_stats.set_background(stats_background)
        self.lbl_word_stats.set_color(text_color)
        self.container_stats.append(self.lbl_word_stats)

        self.lbl_word_raw = ScreenLabel("lbl_word_raw", "Total Raw Words:\n" + str(self.word_total), 21, button_2_width, 1)
        self.lbl_word_raw.position = (button_2_left, self.lbl_word_stats.get_bottom() + 10)
        self.lbl_word_raw.set_background(stats_background)
        self.lbl_word_raw.set_color(text_color)
        self.container_stats.append(self.lbl_word_raw)

        self.lbl_word_unique = ScreenLabel("lbl_word_unique", "Total Unique Words:\n" + str(len(self.unique_groups)), 21, button_2_width, 1)
        self.lbl_word_unique.position = (button_2_right, self.lbl_word_stats.get_bottom() + 10)
        self.lbl_word_unique.set_background(stats_background)
        self.lbl_word_unique.set_color(text_color)
        self.container_stats.append(self.lbl_word_unique)

        #=============================================================
        # Panel with state buttons (Undo, Redo, Save)
        self.container_state_buttons = ScreenContainer("container_state_buttons", (container_width, 250),
                                                       general_background)
        self.container_state_buttons.position = (
        self.container_view_buttons.get_left(), self.container_stats.get_bottom() + 10)
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
        
        self.btn_export = ScreenButton("btn_export", "Export", 21, button_width)
        self.btn_export.set_colors(button_text_color, button_back_color)
        self.btn_export.position = (button_left, self.btn_save.get_bottom() + 10)
        self.btn_export.click_callback = self.btn_export_click
        self.container_state_buttons.append(self.btn_export)

        self.btn_exit = ScreenButton("btn_exit", "Exit", 21, button_width)
        self.btn_exit.set_colors(button_text_color, button_back_color)
        self.btn_exit.position = (button_left, self.btn_export.get_bottom() + 30)
        self.btn_exit.click_callback = self.btn_exit_click
        self.container_state_buttons.append(self.btn_exit)

        # print("MAKE CONTAINER STATS VISIBLE AGAIN!!!")
        # self.container_stats.visible = False
        # self.container_state_buttons.visible = False

        # ==============================================================
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
        #self.img_main.mouse_button_down_callback = self.img_mouse_down
        self.container_images.append(self.img_main)

        # canvas used for annotations
        self.canvas_select = ScreenCanvas("canvas_select", 100, 100)
        self.canvas_select.position = (0, 0)
        self.canvas_select.locked = False
        # self.canvas_select.object_edited_callback = self.canvas_object_edited
        # self.canvas_select.object_selected_callback = self.canvas_selection_changed
        self.container_images.append(self.canvas_select)

        self.canvas_select.add_rectangle_element("selection_rectangle", 10, 10, 40, 40)
        self.canvas_select.elements["selection_rectangle"].visible = False

        self.undo_stack = []
        self.redo_stack = []

        self.update_current_view(True)

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

        # if selection rectangle is active ...
        if self.edition_mode == GTUniqueWordAnnotator.ModeMatch_RegionSelection:
            self.canvas_select.elements["selection_rectangle"].x *= scale_factor
            self.canvas_select.elements["selection_rectangle"].y *= scale_factor
            self.canvas_select.elements["selection_rectangle"].w *= scale_factor
            self.canvas_select.elements["selection_rectangle"].h *= scale_factor

        # update scale text ...
        self.lbl_zoom.set_text("Zoom: " + str(int(round(self.view_scale * 100,0))) + "%")

    def btn_zoom_reduce_click(self, button):
        self.update_view_scale(self.view_scale - 0.25)

    def btn_zoom_increase_click(self, button):
        self.update_view_scale(self.view_scale + 0.25)

    def btn_zoom_zero_click(self, button):
        self.update_view_scale(1.0)

    def translateWord(self, word, disp_x, disp_y):
        return word[0] + disp_x, word[1] + disp_y, word[2], word[3]

    def getBoxesBoundaries(self, bbox):
        # x-min, x-max, y-min, y-max
        return bbox[0], bbox[0] + bbox[2], bbox[1], bbox[1] + bbox[3]

    def getBoxesIOU(self, bbox1, bbox2):
        # get bboxes
        bb1_x_min, bb1_x_max, bb1_y_min, bb1_y_max = self.getBoxesBoundaries(bbox1)
        bb2_x_min, bb2_x_max, bb2_y_min, bb2_y_max = self.getBoxesBoundaries(bbox2)

        # get intersection
        int_min_x = max(bb1_x_min, bb2_x_min)
        int_max_x = min(bb1_x_max, bb2_x_max)
        int_min_y = max(bb1_y_min, bb2_y_min)
        int_max_y = min(bb1_y_max, bb2_y_max)

        # computer intersection area
        int_w = int_max_x - int_min_x
        int_h = int_max_y - int_min_y

        if int_w <= 0.0 or int_h <= 0.0:
            # no IOU!
            return 0.0

        int_area = int_w * int_h

        # get union
        union_min_x = min(bb1_x_min, bb2_x_min)
        union_max_x = max(bb1_x_max, bb2_x_max)
        union_min_y = min(bb1_y_min, bb2_y_min)
        union_max_y = max(bb1_y_max, bb2_y_max)

        # compute union area
        union_w = union_max_x - union_min_x
        union_h = union_max_y - union_min_y

        union_area = union_w * union_h

        IOU = int_area / union_area

        return IOU

    def greedy_matching_scores(self):
        sel_rect = self.canvas_select.elements["selection_rectangle"]
        rect_x = int(round(sel_rect.x / self.view_scale))
        rect_y = int(round(sel_rect.y / self.view_scale))
        rect_w = int(round(sel_rect.w / self.view_scale))
        rect_h = int(round(sel_rect.h / self.view_scale))

        # identify Words from current frame within selected region (containment)
        curr_kf = self.kf_words[self.selected_keyframe]
        curr_words = curr_kf.words_in_region(rect_x, rect_x + rect_w, rect_y, rect_y + rect_h)
        curr_words = {UniqueWordGroup.wordID(word): word for word in curr_words}
        # only keep those that have not been matched yet ...
        filtered_words = {}
        for curr_words_str_id in curr_words:
            if self.word_group[self.selected_keyframe][curr_words_str_id].start_frame == self.selected_keyframe:
                # unmatched CC ...
                filtered_words[curr_words_str_id] = curr_words[curr_words_str_id]

        print("Total candidates (C-KF): " + str(len(curr_words)))
        if len(filtered_words) != len(curr_words):
            print("Total candidates not previously matched (C-KF): " + str(len(filtered_words)))

            curr_words = filtered_words

        # identify CC's from prev frame within selected region (containment)
        prev_kf = self.kf_words[self.selected_keyframe - 1]
        prev_words = prev_kf.words_in_region(rect_x - self.matching_delta_x, rect_x - self.matching_delta_x + rect_w,
                                             rect_y - self.matching_delta_y, rect_y - self.matching_delta_y + rect_h)
        prev_words = {UniqueWordGroup.wordID(word): word for word in prev_words}
        # modify box using delta
        for word_id in prev_words:
            prev_words[word_id] = self.translateWord(prev_words[word_id], self.matching_delta_x, self.matching_delta_y)

        # compute all scores
        all_matches = []
        for curr_word_str_id in curr_words:
            curr_word = curr_words[curr_word_str_id]
            for prev_word_str_id in prev_words:
                prev_word = prev_words[prev_word_str_id]

                IOU = self.getBoxesIOU(curr_word, prev_word)
                if IOU > 0.0:
                    all_matches.append((IOU, prev_word_str_id, curr_word))

        # restore original box
        for word_id in prev_words:
            prev_words[word_id] = self.translateWord(prev_words[word_id], -self.matching_delta_x, -self.matching_delta_y)

        # sort by decreasing recall
        all_matches = sorted(all_matches, reverse=True, key=lambda x: x[0])

        # now, greedily pick 1 to 1 matches ...
        self.matching_scores = []
        matched_curr_ids = {}
        matched_prev_ids = {}
        for IOU, prev_word_str_id, curr_word in all_matches:
            curr_word_str_id = UniqueWordGroup.wordID(curr_word)

            # filter if already matched ....
            if prev_word_str_id in matched_prev_ids:
                continue
            if curr_word_str_id in matched_curr_ids:
                continue

            # accept match ....
            # print((prev_cc, curr_cc))
            self.matching_scores.append((IOU, prev_words[prev_word_str_id], curr_word))

            # marked as matched ...
            matched_prev_ids[prev_word_str_id] = True
            matched_curr_ids[curr_word_str_id] = True

    def btn_confirm_accept_click(self, button):

        if self.edition_mode == GTUniqueWordAnnotator.ModeMatch_RegionSelection:
            # compute potential matches ...
            self.greedy_matching_scores()
            # clear base image where matches will be shown ...
            gray_mask = self.base_matching.sum(axis=2) < 255 * 3
            self.base_matching[gray_mask, 0] = 128
            self.base_matching[gray_mask, 1] = 128
            self.base_matching[gray_mask, 2] = 128
            # move to the next stage ..
            self.set_editor_mode(GTUniqueWordAnnotator.ModeMatch_Matching)
            self.update_current_view(False)

        elif self.edition_mode == GTUniqueWordAnnotator.ModeMatch_Matching:
            # accept matches
            for IOU, prev_word, curr_word in self.matching_scores:
                if IOU >= self.matching_min_IOU:
                    # merge to previous group ...
                    prev_id = UniqueWordGroup.wordID(prev_word)
                    prev_group = self.word_group[self.selected_keyframe - 1][prev_id]

                    curr_id = UniqueWordGroup.wordID(curr_word)
                    curr_group = self.word_group[self.selected_keyframe][curr_id]

                    # for each member of the current group ...
                    for kf_offset, word in enumerate(curr_group.words_refs):
                        # make element point to previous group
                        word_id = UniqueWordGroup.wordID(word)
                        # print(self.word_group[self.selected_keyframe + kf_offset][word_id].start_frame)
                        self.word_group[self.selected_keyframe + kf_offset][word_id] = prev_group
                        # print(self.word_group[self.selected_keyframe + kf_offset][word_id].start_frame)

                        # add element in current group to previous group
                        prev_group.words_refs.append(word)

                    # remove group from list of unique groups
                    self.unique_groups.remove(curr_group)

            # update count ....
            self.lbl_word_unique.set_text("Total Unique Words:\n" + str(len(self.unique_groups)))

            print("PENDING UNDO/REDO")

            self.set_editor_mode(GTUniqueWordAnnotator.ModeNavigate)
            self.update_current_view(False)

        elif self.edition_mode == GTUniqueWordAnnotator.ModeMatch_Remove:
            sel_rect = self.canvas_select.elements["selection_rectangle"]
            rect_x = int(round(sel_rect.x / self.view_scale))
            rect_y = int(round(sel_rect.y / self.view_scale))
            rect_w = int(round(sel_rect.w / self.view_scale))
            rect_h = int(round(sel_rect.h / self.view_scale))

            # identify Words from current frame within selected region (containment)
            curr_kf = self.kf_words[self.selected_keyframe]
            curr_words = curr_kf.words_in_region(rect_x, rect_x + rect_w, rect_y, rect_y + rect_h)
            curr_words = {UniqueWordGroup.wordID(word): word for word in curr_words}

            # only keep those that have been previously matched
            filtered_words = {}
            for word_id in curr_words:
                if self.word_group[self.selected_keyframe][word_id].start_frame < self.selected_keyframe:
                    # matched CC ...
                    filtered_words[word_id] = curr_words[word_id]

            print("Total Words in region (C-KF): " + str(len(curr_words)))
            print("Total matches to remove (C-KF): " + str(len(filtered_words)))
            curr_words = filtered_words

            # Remove Words's from group (split) and add their own group
            for word_id in curr_words:
                # previous group
                prev_group = self.word_group[self.selected_keyframe][word_id]

                # ask the group to split
                new_group = UniqueWordGroup.Split(prev_group, self.selected_keyframe)
                # link CCs on the new group to the new group
                for split_offset, split_word in enumerate(new_group.words_refs):
                    split_word_id = UniqueWordGroup.wordID(split_word)
                    self.word_group[self.selected_keyframe + split_offset][split_word_id] = new_group

                # add new group to the list of unique CC
                self.unique_groups.append(new_group)

            print("Pending to be able to UNDO/REDO")

            self.set_editor_mode(GTUniqueWordAnnotator.ModeNavigate)
            self.lbl_word_unique.set_text("Total Unique Words:\n" + str(len(self.unique_groups)))
            self.update_current_view(False)

    def btn_confirm_cancel_click(self, button):
        # by default, got back to navigation mode ...
        self.set_editor_mode(GTUniqueWordAnnotator.ModeNavigate)
        self.update_current_view(False)

    def btn_view_normal_rgb_click(self, button):
        self.view_mode = GTUniqueWordAnnotator.ViewModeNormalRGB
        self.update_current_view()

    def btn_view_normal_bin_click(self, button):
        self.view_mode = GTUniqueWordAnnotator.ViewModeNormalBin
        self.update_current_view()

    def btn_view_projected_rgb_click(self, button):
        self.view_mode = GTUniqueWordAnnotator.ViewModeProjectedRGB
        self.update_current_view()

    def btn_view_projected_bin_click(self, button):
        self.view_mode = GTUniqueWordAnnotator.ViewModeProjectedBin
        self.update_current_view()

    def prepare_selection_rectangle(self, margin):
        # Default selection rectangle is relative to current view
        if self.container_images.v_scroll.active:
            rect_y = margin + self.container_images.v_scroll.value
            rect_h = self.container_images.height - (margin * 2) - self.container_images.h_scroll.height
        else:
            rect_y = margin
            rect_h = self.img_main.height - margin * 2

        if self.container_images.h_scroll.active:
            rect_x = margin + self.container_images.h_scroll.value
            rect_w = self.container_images.width - margin * 2 - self.container_images.v_scroll.width
        else:
            rect_x = margin
            rect_w = self.img_main.width - margin * 2

        self.canvas_select.elements["selection_rectangle"].x = rect_x
        self.canvas_select.elements["selection_rectangle"].y = rect_y
        self.canvas_select.elements["selection_rectangle"].w = rect_w
        self.canvas_select.elements["selection_rectangle"].h = rect_h

    def btn_matches_add_click(self, button):
        if self.selected_keyframe == 0:
            print("Cannot match elements on the first frame")
            return

        self.prepare_selection_rectangle(40)
        self.set_editor_mode(GTUniqueWordAnnotator.ModeMatch_RegionSelection)
        self.update_matching_image()
        self.update_current_view(False)


    def btn_matches_del_click(self, button):
        if self.selected_keyframe == 0:
            print("There are no matches on the first frame")
            return

        self.prepare_selection_rectangle(40)
        self.set_editor_mode(GTUniqueWordAnnotator.ModeMatch_Remove)
        self.update_current_view(False)

    def btn_undo_click(self, button):
        print("Not yet available!")
        pass

    def btn_redo_click(self, button):
        print("Not yet available!")
        pass

    def btn_save_click(self, button):
        xml_str = UniqueWordGroup.GenerateGroupsXML(self.kf_words, self.unique_groups)

        unique_words_filename = self.output_path + "/unique_words.xml"
        out_file = open(unique_words_filename, "w")
        out_file.write(xml_str)
        out_file.close()

        print("Saved to: " + unique_words_filename)

    def btn_exit_click(self, button):
        if len(self.undo_stack) > 0:
            # confirm before losing changes
            self.set_editor_mode(GTUniqueWordAnnotator.ModeExitConfirm)
        else:
            # Just exit
            self.return_screen = None
            print("APPLICATION FINISHED")

    def update_matching_image(self):
        prev_binary = self.keyframe_annotations[self.selected_keyframe - 1].binary_image[:, :, 0]
        curr_binary = self.keyframe_annotations[self.selected_keyframe].binary_image[:, :, 0]

        self.base_matching = Visualizer.combine_bin_images_w_disp(curr_binary, prev_binary, self.matching_delta_x,
                                                                  self.matching_delta_y, 0)

    def scroll_delta_x_change(self, scroll):
        self.matching_delta_x = int(scroll.value)
        self.lbl_delta_x.set_text("Delta X: " + str(self.matching_delta_x))
        self.update_matching_image()
        self.update_current_view()

    def scroll_delta_y_change(self, scroll):
        self.matching_delta_y = int(scroll.value)
        self.lbl_delta_y.set_text("Delta Y: " + str(self.matching_delta_y))
        self.update_matching_image()
        self.update_current_view()

    def scroll_min_IOU_change(self, scroll):
        self.matching_min_IOU = scroll.value / 100.0
        self.lbl_min_IOU.set_text("Minimum IOU: " + str(int(self.matching_min_IOU * 100)))
        self.update_current_view()

    def set_editor_mode(self, new_mode):
        self.edition_mode = new_mode
        self.container_nav_buttons.visible = (new_mode == GTUniqueWordAnnotator.ModeNavigate)
        # self.container_view_buttons.visible = (new_mode == GTUniqueWordAnnotator.ModeNavigate)
        self.btn_view_normal_bin.visible = (new_mode == GTUniqueWordAnnotator.ModeNavigate)
        self.btn_view_normal_rgb.visible = (new_mode == GTUniqueWordAnnotator.ModeNavigate)
        self.btn_view_projected_bin.visible = (new_mode == GTUniqueWordAnnotator.ModeNavigate)
        self.btn_view_projected_rgb.visible = (new_mode == GTUniqueWordAnnotator.ModeNavigate)

        self.container_confirm_buttons.visible = (new_mode == GTUniqueWordAnnotator.ModeMatch_RegionSelection or
                                                  new_mode == GTUniqueWordAnnotator.ModeMatch_Matching or
                                                  new_mode == GTUniqueWordAnnotator.ModeMatch_Remove or
                                                  new_mode == GTUniqueWordAnnotator.ModeExitConfirm)

        if new_mode == GTUniqueWordAnnotator.ModeMatch_RegionSelection:
            self.lbl_confirm_message.set_text("Selecting Matching Region")
        elif new_mode == GTUniqueWordAnnotator.ModeMatch_Matching:
            self.lbl_confirm_message.set_text("Matching Unique CCs")
        elif new_mode == GTUniqueWordAnnotator.ModeMatch_Remove:
            self.lbl_confirm_message.set_text("Remove CCs Matches")
        elif new_mode == GTUniqueWordAnnotator.ModeExitConfirm:
            self.lbl_confirm_message.set_text("Exit Without Saving?")

        if new_mode == GTUniqueWordAnnotator.ModeMatch_RegionSelection or new_mode == GTUniqueWordAnnotator.ModeMatch_Remove:
            # show rectangle
            self.canvas_select.locked = False
            self.canvas_select.elements["selection_rectangle"].visible = True
        else:
            # for every other mode
            self.canvas_select.locked = True
            self.canvas_select.elements["selection_rectangle"].visible = False

        self.container_state_buttons.visible = (new_mode == GTUniqueWordAnnotator.ModeNavigate)
        self.container_stats.visible = (new_mode == GTUniqueWordAnnotator.ModeNavigate)
        self.container_action_buttons.visible = (new_mode == GTUniqueWordAnnotator.ModeNavigate)

        self.container_matching_translation.visible = (new_mode == GTUniqueWordAnnotator.ModeMatch_RegionSelection)
        self.container_matching_strictness.visible = (new_mode == GTUniqueWordAnnotator.ModeMatch_Matching)

    def get_segments_from_annotation(self, root, namespace):
        # load video segments ...
        xml_video_segments_root = root.find(namespace + "VideoSegments")
        xml_video_segment_objects = xml_video_segments_root.findall(namespace + "VideoSegment")
        segments = []
        for xml_video_segment_object in xml_video_segment_objects:
            start_point = int(xml_video_segment_object.find(VideoObject.XMLNamespace + 'Start').text)
            end_point = int(xml_video_segment_object.find(VideoObject.XMLNamespace + 'End').text)
            segments.append((start_point, end_point))

        return segments

    def btn_export_click(self, button):
        input_filename = self.output_path + ".xml"
        output_filename = self.output_path + "_words.xml"

        nmspace = ''
        prefix = "word_"
        raw_h, raw_w, _ = self.keyframe_annotations[0].raw_image.shape
        xml_root = ET.parse(input_filename)

        # objects_root = xml_root.find(nmspace + "VideoObjects")

        # need segments for timing of objects ...
        segments = self.get_segments_from_annotation(xml_root, nmspace)

        kf_segments = {}
        kf_times = []
        kf_indices = []
        for idx, kf in enumerate(self.keyframe_annotations):
            segment_idx = 0
            while segments[segment_idx][1] < self.keyframe_annotations[idx].idx:
                segment_idx += 1

            kf_segments[idx] = segment_idx
            kf_times.append(self.keyframe_annotations[idx].time)
            kf_indices.append(self.keyframe_annotations[idx].idx)

        f_frame_time = interpolate.interp1d(kf_indices, kf_times, fill_value="extrapolate")

        # need canvas info .. project word locations ...
        draw_info = DrawingInfo.from_XML(xml_root, nmspace)

        # create object for unique words ...
        n_zeros = int(math.ceil(math.log(len(self.unique_groups) + 1, 10)))
        all_hw_video_objects = []
        for word_idx, word_group in enumerate(self.unique_groups):
            assert isinstance(word_group, UniqueWordGroup)

            word_name = prefix + str(word_idx).zfill(n_zeros)
            word_object = VideoObject(word_name, word_name, VideoObject.ShapeQuadrilateral)

            # determine key-frames for object ...
            obj_kf_idxs = []
            obj_kf_polygons = {}
            n_frames = word_group.n_frames()
            for rel_offset, word_frame_idx in enumerate(range(word_group.start_frame, word_group.lastFrame() + 1)):
                seg_start, seg_end = segments[kf_segments[word_frame_idx]]
                current_frame_idx = self.keyframe_annotations[word_frame_idx].idx
                current_frame_time = self.keyframe_annotations[word_frame_idx].time
                current_projection = self.kf_words[word_frame_idx].projection

                # create corresponding polygon ...
                frame_bbox = word_group.words_refs[rel_offset]
                # from BBOX to polygon BBOX
                frame_polygons = current_projection.bboxesToPolygons([frame_bbox])
                # warp the polygon based on the projection ...
                image_polygon = current_projection.warpPolygons(frame_polygons, True)[0]
                # then, project from image space to editor space ...
                vol_loc_polygon = draw_info.unproject_polygon(raw_w, raw_h, image_polygon)

                obj_kf_polygons[current_frame_idx] = vol_loc_polygon

                if rel_offset == 0:
                    # add first
                    obj_kf_idxs.append((seg_start, None, f_frame_time([seg_start])[0]))
                # add current
                obj_kf_idxs.append((current_frame_idx, current_frame_idx, current_frame_time))

                if rel_offset == n_frames - 1:
                    # add last
                    obj_kf_idxs.append((seg_end, None, f_frame_time([seg_end])[0]))

            # obj_kf_idxs = sorted(list(obj_kf_idxs), key=lambda x:x[0])

            # add the locations ...
            for offset, (frame_idx, parent_frame_idx, frame_time) in enumerate(obj_kf_idxs):
                # check if first ..
                if parent_frame_idx is None:
                    # special case
                    if offset == 0:
                        # first ... copy polygon from next frame ...
                        ref_frame_idx = obj_kf_idxs[offset + 1][0]
                    else:
                        # last ... copy polygon from prev frame ...
                        ref_frame_idx = obj_kf_idxs[offset - 1][0]

                    current_polygon = obj_kf_polygons[ref_frame_idx]
                else:
                    # normal case
                    current_polygon = obj_kf_polygons[parent_frame_idx]

                word_object.set_location_at(frame_idx, frame_time, True, current_polygon)

            # add to temporal list ...
            all_hw_video_objects.append(word_object)

        # now add to the original annotations
        objects_root = xml_root.find("VideoObjects")
        for hw_video_object in all_hw_video_objects:
            # convert data to XML ...
            hw_xml_node = ET.fromstring(hw_video_object.toXML())
            # ... add to the tree
            objects_root.append(hw_xml_node)

        xml_root.write(output_filename)

        print("Data exported to: " + output_filename)

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

    def update_current_view(self, resized=False):
        if (self.edition_mode == GTUniqueWordAnnotator.ModeMatch_RegionSelection or
            self.edition_mode == GTUniqueWordAnnotator.ModeMatch_Matching):
            # special override case, override current view to show matching image
            base_image = self.base_matching
        elif self.edition_mode == GTUniqueWordAnnotator.ModeMatch_Remove:
            base_image = self.keyframe_annotations[self.selected_keyframe].binary_image
        elif self.view_mode == GTUniqueWordAnnotator.ViewModeNormalBin:
            # the binary after projected on the original space ...
            base_image = self.unprojected_BIN_cache[self.selected_keyframe]
        elif self.view_mode == GTUniqueWordAnnotator.ViewModeProjectedRGB:
            # currently keeps a copy of the projected image ...
            base_image = self.keyframe_annotations[self.selected_keyframe].raw_image
        elif self.view_mode == GTUniqueWordAnnotator.ViewModeProjectedBin:
            # binary generated in projected space ...
            base_image = self.keyframe_annotations[self.selected_keyframe].binary_image
        else:
            # default for ViewModeNormalRGB ... original before projection
            base_image = self.unprojected_RGB_cache[self.selected_keyframe]


        h, w, c = base_image.shape
        modified_image = base_image.copy()

        projection = self.kf_words[self.selected_keyframe].projection

        if self.edition_mode == GTUniqueWordAnnotator.ModeMatch_Matching:
            # color the affected words by the current selection

            # find matches/unmatches
            matching_bboxes = []
            unmatched_bboxes = []
            for IOU, prev_word, curr_word in self.matching_scores:
                if IOU >= self.matching_min_IOU:
                    matching_bboxes.append(curr_word)
                else:
                    unmatched_bboxes.append(curr_word)

            # convert to polygons
            matched_polygons = projection.bboxesToPolygons(matching_bboxes)
            unmatched_polygons = projection.bboxesToPolygons(unmatched_bboxes)

            # prepare for drawing
            reshaped_matched = []
            reshaped_unmatched = []
            for polygon in matched_polygons:
                reshaped_matched.append(polygon.reshape((-1, 1, 2)).astype(np.int32))

            for polygon in unmatched_polygons:
                reshaped_unmatched.append(polygon.reshape((-1, 1, 2)).astype(np.int32))

            # Draw the polygons
            cv2.polylines(modified_image, reshaped_matched, True, (0, 255, 0), thickness=2)
            cv2.polylines(modified_image, reshaped_unmatched, True, (255, 0, 0), thickness=2)

        else:
            # draw boxes on current visible frame
            # get the boxes and convert to polygon
            # TODO: cache this step as much as it can be cached
            current_bboxes = self.collected_words[self.selected_keyframe]

            polygons = projection.bboxesToPolygons(current_bboxes)

            if (self.view_mode == GTUniqueWordAnnotator.ViewModeProjectedRGB or
                self.view_mode == GTUniqueWordAnnotator.ViewModeProjectedBin or
                self.edition_mode == GTUniqueWordAnnotator.ModeMatch_RegionSelection or
                self.edition_mode == GTUniqueWordAnnotator.ModeMatch_Remove):
                # use the original bboxes ... no projections needed in this space
                current_polygons = polygons
            else:
                current_polygons = projection.warpPolygons(polygons, True)

            # group polygons per color ...
            n_colors = len(GTUniqueWordAnnotator.WordsShowColors)
            per_color_polygons = {}
            for idx, word in enumerate(current_bboxes):
                start = self.word_group[self.selected_keyframe][UniqueWordGroup.wordID(word)].start_frame
                color_idx = start % n_colors

                draw_polygon = current_polygons[idx].reshape((-1, 1, 2)).astype(np.int32)

                if color_idx in per_color_polygons:
                    per_color_polygons[color_idx].append(draw_polygon)
                else:
                    per_color_polygons[color_idx] = [draw_polygon]

            for color_idx in per_color_polygons:
                word_color = GTUniqueWordAnnotator.WordsShowColors[color_idx]
                cv2.polylines(modified_image, per_color_polygons[color_idx], True, word_color, thickness=2)

        # finally, resize ...
        modified_image = cv2.resize(modified_image, (int(w * self.view_scale), int(h * self.view_scale)),
                                    interpolation=cv2.INTER_NEAREST)

        self.canvas_select.height, self.canvas_select.width, _ = modified_image.shape

        # replace/update image
        self.img_main.set_image(modified_image, 0, 0, True, cv2.INTER_NEAREST)
        if resized:
            self.container_images.recalculate_size()
