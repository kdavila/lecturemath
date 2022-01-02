
import os
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np


from AM_CommonTools.util.time_helper import TimeHelper
from AM_CommonTools.interface.controls.screen import Screen
from AM_CommonTools.interface.controls.screen_button import ScreenButton
from AM_CommonTools.interface.controls.screen_canvas import ScreenCanvas
from AM_CommonTools.interface.controls.screen_container import ScreenContainer
from AM_CommonTools.interface.controls.screen_horizontal_scroll import ScreenHorizontalScroll
from AM_CommonTools.interface.controls.screen_vertical_scroll import ScreenVerticalScroll
from AM_CommonTools.interface.controls.screen_label import ScreenLabel
from AM_CommonTools.interface.controls.screen_textbox import ScreenTextbox
from AM_CommonTools.interface.controls.screen_textlist import ScreenTextlist
from AM_CommonTools.interface.controls.screen_video_player import ScreenVideoPlayer
from AM_CommonTools.interface.controls.screen_timer import ScreenTimer

from AccessMath.annotation.drawing_info import DrawingInfo
from AccessMath.annotation.video_object import VideoObject
from AccessMath.annotation.video_object_location import VideoObjectLocation
from AccessMath.annotation.lecture_annotation import LectureAnnotation


class GTContentAnnotator(Screen):
    EditGroupingTime = 5.0

    def __init__(self, size, video_metadadata, db_name, lecture_title, output_prefix, forced_resolution=None):
        Screen.__init__(self, "Ground Truth Annotation Interface", size)

        self.general_background = (80, 80, 95)
        self.elements.back_color = self.general_background

        # self.db_name = db_name
        # self.lecture_title = lecture_title

        self.output_prefix = output_prefix
        self.output_filename = output_prefix + ".xml"

        # check input video metadata  ...
        all_paths = []
        sequence_mode = False
        self.img_format = None
        for video in video_metadadata:
            all_paths.append(video["path"])
            if video["type"].lower() == "imagelist":
                sequence_mode = True
                self.img_format = "." + video["format"]

        self.video_files = all_paths
        self.forced_resolution = forced_resolution

        self.user_selected_position = None
        self.user_last_name_prefix = None

        # main video player
        self.player_type = (ScreenVideoPlayer.VideoPlayerImageList if sequence_mode
                            else ScreenVideoPlayer.VideoPlayerOpenCV)

        # Base annotation elements
        self.player = None
        self.canvas = None
        # self.label_title = None
        self.exit_button = None

        self.last_video_frame = None
        self.last_video_time = None

        # video controllers
        self.container_video_controls = None
        self.position_scroll = None
        self.label_frame_count = None
        self.label_frame_current = None
        self.label_time_current = None
        self.label_player_speed = None
        self.btn_dec_speed = None
        self.btn_inc_speed = None
        self.precision_buttons = None
        self.button_pause = None
        self.button_play = None
        self.panning_hor_scroll = None
        self.panning_ver_scroll = None
        self.btn_inc_zoom = None
        self.btn_dec_zoom = None
        self.label_zoom_current = None


        # Container 1: Object selector and controls ...
        self.container_object_options = None
        self.object_selector = None
        self.label_object_selector = None
        self.btn_object_add_bbox = None
        self.btn_object_add_quad = None
        self.btn_object_rename = None
        self.btn_object_remove = None
        self.btn_object_split_poly = None

        # ... key-frame buttons ...
        self.container_keyframe_options = None
        self.lbl_keyframe_title = None
        self.btn_keyframe_invisible = None
        self.btn_keyframe_visible = None
        self.btn_keyframe_prev = None
        self.btn_keyframe_next = None
        self.btn_keyframe_add = None
        self.btn_keyframe_del = None
        self.lbl_keyframe_prev = None
        self.lbl_keyframe_next = None
        self.lbl_keyframe_label = None
        self.btn_keyframe_label_set = None

        # ... Video Segments buttons ...
        self.container_vid_seg_options = None
        self.lbl_vid_seg_title = None
        self.btn_vid_seg_prev = None
        self.btn_vid_seg_next = None
        self.btn_vid_seg_split = None
        self.btn_vid_seg_merge = None
        self.lbl_vid_seg_prev = None
        self.lbl_vid_seg_next = None

        # ... Video Segments Keyframes buttons ...
        self.container_vid_seg_keyframe_options = None
        self.lbl_vid_seg_keyframe_title = None
        self.btn_vid_seg_keyframe_prev = None
        self.btn_vid_seg_keyframe_next = None
        self.btn_vid_seg_keyframe_add = None
        self.btn_vid_seg_keyframe_del = None
        self.lbl_vid_seg_keyframe_prev = None
        self.lbl_vid_seg_keyframe_next = None

        # Container for text input ....
        self.container_text_input = None
        self.txt_object_name = None
        self.label_txt_object_name = None
        self.btn_text_operation_accept = None
        self.btn_text_operation_cancel = None
        self.label_txt_object_error = None

        self.on_forced_exit = self.forced_exit_clicked

        # =============================================
        # special container for buttons with options for quick labeling
        self.container_segment_labels = []
        self.buttons_prev_segment_labels = []

        # ===============================
        #  Other general buttons and controls
        self.save_button = None
        self.export_button = None
        self.export_object_button = None
        self.redo_button = None
        self.undo_button = None
        self.auto_save_timer = None

        self.create_base_controllers()
        self.create_video_controls()
        self.create_object_options()
        self.create_keyframes_buttons()
        self.create_video_segments_buttons()
        self.create_video_segments_keyframes_buttons()
        self.create_text_input_buttons()
        self.create_segment_labels_buttons()
        self.create_general_buttons()

        # 0 - Nothing
        # 1 - Adding new Object
        # 2 - Renaming Object
        # 3 - Confirm Deleting Object
        # 4 - Confirm Exit without saving
        self.text_operation = 0
        # Based on VideoObject Shapes
        self.new_object_shape = 0
        self.changes_saved = True
        self.undo_stack = []
        self.redo_stack = []

        canvas_bbox = (self.canvas.position[0], self.canvas.position[1], self.canvas.width, self.canvas.height)
        player_bbox = (self.player.position[0], self.player.position[1], self.player.width, self.player.height)
        render_bbox = (self.player.render_location[0], self.player.render_location[1],
                       self.player.render_width, self.player.render_height)

        drawing_info = DrawingInfo(canvas_bbox, player_bbox, render_bbox)

        # Load/Create annotation file
        if os.path.exists(self.output_filename):
            print("Saved file exists. Loading ...")
            self.lecture = LectureAnnotation.Load(self.output_filename, True)

            if not drawing_info == self.lecture.drawing_info:
                print("Original Drawing parameters")
                print(self.lecture.drawing_info)
                print("Current Drawing parameters")
                print(drawing_info)

                if not drawing_info.equivalent_areas(self.lecture.drawing_info):
                    raise Exception("Cannot load/save annotation files created with different drawing parameters (yet)")
                else:
                    # the player and canvas have the same size, but they have been moved in the GUI
                    # this should not affect the relative coordinate system used in the file
                    print("Warning: The drawing information has changed. It will be updated on save")
                    # ... update ...
                    self.lecture.drawing_info = drawing_info

            if not self.video_files == self.lecture.video_files:
                print("Warning: Original annotation video files and current video files do not match!")
                print("Video Files on Annotation")
                print(self.lecture.video_files)
                print("Current Video Files:")
                print(self.video_files)
                print("")
                print("Original video file names will be overridden on saving")
                self.lecture.video_files = self.video_files

            if not self.player.video_player.total_frames == self.lecture.total_frames:
                print("Warning: Total number of frames on annotation and current video files do not match!")
                print("Annotation: " + str(self.lecture.total_frames))
                print("Current: " + str(self.player.video_player.total_frames))
                print("")
                print("Original video length will be overridden on saving")
                self.lecture.total_frames = self.player.video_player.total_frames

            self.load_saved_data_into_GUI()

        else:
            total_frames = self.player.video_player.total_frames
            self.lecture = LectureAnnotation(db_name, lecture_title, self.output_filename, self.video_files,
                                             total_frames, drawing_info)

        self.lecture.set_frame_resolution(self.player.video_player.width, self.player.video_player.height)

        self.update_video_segment_buttons()

        self.elements.key_up_callback = self.main_key_up

    def create_base_controllers(self):
        # main video player
        self.player = ScreenVideoPlayer("video_player", 960, 540)
        # self.player.position = (50, 100)
        self.player.position = (50, 50)
        self.player.open_video_files(self.video_files, self.forced_resolution, self.player_type, self.img_format)
        self.player.frame_changed_callback = self.video_frame_change
        self.player.click_with_pos_callback = self.player_click
        self.player.play()


        self.elements.append(self.player)

        print("Total Video Length: " + TimeHelper.secondsToStr(self.player.video_player.total_length))
        print("Total Video Frames: " + str(self.player.video_player.total_frames))

        # canvas used for annotations
        self.canvas = ScreenCanvas("canvas", 1040, 620)
        # self.canvas.position = (10, 60)
        self.canvas.position = (10, 10)
        self.canvas.locked = True
        self.canvas.object_edited_callback = self.canvas_object_edited
        self.canvas.object_selected_callback = self.canvas_selection_changed
        self.elements.append(self.canvas)

        # add elements....
        # TITLE
        # self.label_title = ScreenLabel("title", "ACCESS MATH - Video Annotation Tool", 28)
        # self.label_title.background = self.general_background
        # self.label_title.position = (int((self.width - self.label_title.width) / 2), 20)
        # self.label_title.set_color((255, 255, 255))
        # self.elements.append(self.label_title)

        # EXIT BUTTON
        self.exit_button = ScreenButton("exit_button", "EXIT", 16, 70, 0)
        self.exit_button.set_colors((192, 255, 128), (64, 64, 64))
        self.exit_button.position = (self.width - self.exit_button.width - 15,
                                     self.height - self.exit_button.height - 15)
        self.exit_button.click_callback = self.close_click
        self.elements.append(self.exit_button)

        self.auto_save_timer = ScreenTimer("auto_save_timer", 300, True)
        self.auto_save_timer.timer_callback = self.auto_save_timer_tick
        self.elements.append(self.auto_save_timer)

    def create_video_controls(self):
        self.container_video_controls = ScreenContainer("container_video_controls", (1050, 200),
                                                        self.general_background)
        self.container_video_controls.position = (5, self.canvas.get_bottom() + 5)
        self.elements.append(self.container_video_controls)

        # zoom in controls
        self.panning_hor_scroll = ScreenHorizontalScroll("panning_hor_scroll", 0, 100, 0, 10)
        self.panning_hor_scroll.position = (5, 5)
        self.panning_hor_scroll.width = 1040
        self.panning_hor_scroll.scroll_callback = self.panning_hor_scroll_change
        self.panning_hor_scroll.visible = False
        self.container_video_controls.append(self.panning_hor_scroll)

        self.panning_ver_scroll = ScreenVerticalScroll("panning_ver_scroll", 0, 100, 0, 10)
        self.panning_ver_scroll.position = (self.canvas.get_right() + 15, self.canvas.get_top())
        self.panning_ver_scroll.height = self.canvas.height
        self.panning_ver_scroll.scroll_callback = self.panning_ver_scroll_change
        self.panning_ver_scroll.visible = False
        self.elements.append(self.panning_ver_scroll)

        # Frame count
        self.label_frame_count = ScreenLabel("frame_count",
                                             "Frame Count: " + str(int(self.player.video_player.total_frames)), 18)
        self.label_frame_count.position = (15, self.panning_hor_scroll.get_bottom() + 30)
        self.label_frame_count.set_color((255, 255, 255))
        self.label_frame_count.set_background((80, 80, 95))
        self.container_video_controls.append(self.label_frame_count)

        # Current Frame
        self.label_frame_current = ScreenLabel("frame_current", "Current Frame: 0", 18)
        self.label_frame_current.position = (175, int(self.label_frame_count.get_top()))
        self.label_frame_current.set_color((255, 255, 255))
        self.label_frame_current.set_background((80, 80, 95))
        self.container_video_controls.append(self.label_frame_current)

        # Current Time
        self.label_time_current = ScreenLabel("time_current", "Current Time: 0", 18)
        self.label_time_current.position = (175, int(self.label_frame_current.get_bottom() + 15))
        self.label_time_current.set_color((255, 255, 255))
        self.label_time_current.set_background((80, 80, 95))
        self.container_video_controls.append(self.label_time_current)

        # player speed
        self.label_player_speed = ScreenLabel("label_player_speed", "Speed: 100%", 18)
        self.label_player_speed.position = (475, int(self.label_frame_count.get_top()))
        self.label_player_speed.set_color((255, 255, 255))
        self.label_player_speed.set_background((80, 80, 95))
        self.container_video_controls.append(self.label_player_speed)

        # Player speed buttons
        self.btn_dec_speed = ScreenButton("btn_dec_speed", "0.5x", 16, 70, 0)
        self.btn_dec_speed.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_dec_speed.position = (self.label_player_speed.get_left() - self.btn_dec_speed.width - 15,
                                       self.label_player_speed.get_top())
        self.btn_dec_speed.click_callback = self.btn_dec_speed_click
        self.container_video_controls.append(self.btn_dec_speed)

        self.btn_inc_speed = ScreenButton("btn_inc_speed", "2.0x", 16, 70, 0)
        self.btn_inc_speed.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_inc_speed.position = (self.label_player_speed.get_right() + 15, self.label_player_speed.get_top())
        self.btn_inc_speed.click_callback = self.btn_inc_speed_click
        self.container_video_controls.append(self.btn_inc_speed)

        # Player Zoom
        self.label_zoom_current = ScreenLabel("label_zoom_current", "Zoom: 100%", 18)
        self.label_zoom_current.position = (775, int(self.label_frame_count.get_top()))
        self.label_zoom_current.set_color((255, 255, 255))
        self.label_zoom_current.set_background((80, 80, 95))
        self.container_video_controls.append(self.label_zoom_current)

        # Player Zoom buttons
        self.btn_dec_zoom = ScreenButton("btn_dec_zoom", "0.5x", 16, 70, 0)
        self.btn_dec_zoom.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_dec_zoom.position = (self.label_zoom_current.get_left() - self.btn_dec_zoom.width - 15,
                                       self.label_zoom_current.get_top())
        self.btn_dec_zoom.click_callback = self.btn_dec_zoom_click
        self.container_video_controls.append(self.btn_dec_zoom)

        self.btn_inc_zoom = ScreenButton("btn_inc_zoom", "2.0x", 16, 70, 0)
        self.btn_inc_zoom.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_inc_zoom.position = (self.label_zoom_current.get_right() + 15, self.label_zoom_current.get_top())
        self.btn_inc_zoom.click_callback = self.btn_inc_zoom_click
        self.container_video_controls.append(self.btn_inc_zoom)

        step_1 = self.player.video_player.total_frames / 100
        self.position_scroll = ScreenHorizontalScroll("video_position", 0, self.player.video_player.total_frames - 1, 0,
                                                      step_1)
        self.position_scroll.position = (5, self.label_time_current.get_bottom() + 10)
        self.position_scroll.width = 1040
        self.position_scroll.scroll_callback = self.main_scroll_change
        self.container_video_controls.append(self.position_scroll)

        # Precision buttons ....
        v_pos = self.position_scroll.get_bottom() + 15
        btn_w = 70
        self.precision_buttons = {}
        for idx, value in enumerate([-1000, -100, -10, -1]):
            prec_button = ScreenButton("prec_button_m_" + str(idx), str(value), 16, btn_w, 0)
            prec_button.set_colors((192, 255, 128), (64, 64, 64))
            prec_button.position = (15 + idx * (btn_w + 15), v_pos)
            prec_button.click_callback = self.btn_change_frame
            prec_button.tag = value
            self.container_video_controls.append(prec_button)
            self.precision_buttons[value] = prec_button

        self.button_pause = ScreenButton("btn_pause", "Pause", 16, 70, 0)
        self.button_pause.set_colors((192, 255, 128), (64, 64, 64))
        self.button_pause.position = (15 + 4 * (btn_w + 15), v_pos)
        self.button_pause.click_callback = self.btn_pause_click
        self.container_video_controls.append(self.button_pause)

        self.button_play = ScreenButton("btn_play", "Play", 16, 70, 0)
        self.button_play.set_colors((192, 255, 128), (64, 64, 64))
        self.button_play.position = (15 + 4 * (btn_w + 15), v_pos)
        self.button_play.click_callback = self.btn_play_click
        self.button_play.visible = False
        self.container_video_controls.append(self.button_play)

        for idx, value in enumerate([1, 10, 100, 1000]):
            prec_button = ScreenButton("prec_button_p_" + str(idx), str(value), 16, btn_w, 0)
            prec_button.set_colors((192, 255, 128), (64, 64, 64))
            prec_button.position = (15 + (5 + idx) * (btn_w + 15), v_pos)
            prec_button.click_callback = self.btn_change_frame
            prec_button.tag = value
            self.container_video_controls.append(prec_button)
            self.precision_buttons[value] = prec_button

    def create_object_options(self):
        self.container_object_options = ScreenContainer("container_object_options", (425, 390), self.general_background)
        self.container_object_options.position = (self.width - self.container_object_options.width - 20,
                                                    self.canvas.get_top())

        # ... Object selector ...
        self.object_selector = ScreenTextlist("object_selector", (295, 340), 21, (40, 40, 48), (255, 255, 255),
                                              (190, 190, 128), (0, 0, 0))
        self.object_selector.position = (10, 40)
        self.object_selector.selected_value_change_callback = self.object_selector_option_changed
        self.container_object_options.append(self.object_selector)
        # ... label ...
        self.label_object_selector = ScreenLabel("label_object_selector", "Video Objects", 26)
        self.label_object_selector.background = self.general_background
        tempo_x_pos = self.object_selector.get_center_x() - self.label_object_selector.width / 2
        self.label_object_selector.position = (tempo_x_pos, 5)
        self.label_object_selector.set_color((255, 255, 255))
        self.container_object_options.append(self.label_object_selector)

        # ... object buttons ....
        # ...... add a bounding box (Axis Aligned Rectangle) ....
        self.btn_object_add_bbox = ScreenButton("btn_object_add_bbox", "+ BBOX", 22, 100)
        self.btn_object_add_bbox.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_object_add_bbox.position = (self.container_object_options.width - self.btn_object_add_bbox.width - 5,
                                             self.object_selector.get_top())
        self.btn_object_add_bbox.click_callback = self.btn_object_add_bbox_click
        self.container_object_options.append(self.btn_object_add_bbox)
        # ...... add a quadrilateral (polygon, sides 4)
        self.btn_object_add_quad = ScreenButton("btn_object_add_quad", "+ QUAD", 22, 100)
        self.btn_object_add_quad.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_object_add_quad.position = (self.container_object_options.width - self.btn_object_add_quad.width - 5,
                                             self.btn_object_add_bbox.get_bottom() + 15)
        self.btn_object_add_quad.click_callback = self.btn_object_add_quad_click
        self.container_object_options.append(self.btn_object_add_quad)

        # ...... rename ....
        self.btn_object_rename = ScreenButton("btn_object_rename", "Rename", 22, 100)
        self.btn_object_rename.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_object_rename.position = (self.container_object_options.width - self.btn_object_rename.width - 5,
                                           self.btn_object_add_quad.get_bottom() + 15)
        self.btn_object_rename.click_callback = self.btn_object_rename_click
        self.container_object_options.append(self.btn_object_rename)

        # ...... remove ....
        self.btn_object_remove = ScreenButton("btn_object_remove", "Remove", 22, 100)
        self.btn_object_remove.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_object_remove.position = (self.container_object_options.width - self.btn_object_remove.width - 5,
                                           self.btn_object_rename.get_bottom() + 15)
        self.btn_object_remove.click_callback = self.btn_object_remove_click
        self.container_object_options.append(self.btn_object_remove)

        # ...... split quad / poly .....
        self.btn_object_split_poly = ScreenButton("btn_object_split_poly", "Split", 22, 100)
        self.btn_object_split_poly.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_object_split_poly.position = (self.container_object_options.width - self.btn_object_split_poly.width - 5,
                                               self.btn_object_remove.get_bottom() + 55)
        self.btn_object_split_poly.click_callback = self.btn_object_split_poly_click
        self.btn_object_split_poly.visible = False
        self.container_object_options.append(self.btn_object_split_poly)

        self.elements.append(self.container_object_options)

    def create_keyframes_buttons(self):
        self.container_keyframe_options = ScreenContainer("container_keyframe_options", (425, 150),
                                                          self.general_background)
        self.container_keyframe_options.position = (self.width - self.container_keyframe_options.width - 20,
                                                    self.container_object_options.get_bottom() + 5)
        self.container_keyframe_options.visible = False

        # lbl_keyframe_title
        self.lbl_keyframe_title = ScreenLabel("lbl_keyframe_title", "Object Key-frames: ", 21, 415, 1)
        self.lbl_keyframe_title.set_color((255, 255, 255))
        self.lbl_keyframe_title.set_background(self.general_background)
        self.lbl_keyframe_title.position = (5, 5)
        self.container_keyframe_options.append(self.lbl_keyframe_title)

        # btn_keyframe_invisible
        self.btn_keyframe_invisible = ScreenButton("btn_keyframe_invisible", "Hide", 22, 75)
        self.btn_keyframe_invisible.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_keyframe_invisible.position = (25, self.lbl_keyframe_title.get_bottom() + 10)
        self.btn_keyframe_invisible.click_callback = self.btn_keyframe_invisible_click
        self.container_keyframe_options.append(self.btn_keyframe_invisible)

        # btn_keyframe_visible
        self.btn_keyframe_visible = ScreenButton("btn_keyframe_visible", "Show", 22, 75)
        self.btn_keyframe_visible.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_keyframe_visible.position = (25, self.lbl_keyframe_title.get_bottom() + 10)
        self.btn_keyframe_visible.click_callback = self.btn_keyframe_visible_click
        self.container_keyframe_options.append(self.btn_keyframe_visible)

        # btn_keyframe_prev
        self.btn_keyframe_prev = ScreenButton("btn_keyframe_prev", "Prev", 22, 75)
        self.btn_keyframe_prev.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_keyframe_prev.position = (125, self.lbl_keyframe_title.get_bottom() + 10)
        self.btn_keyframe_prev.click_callback = self.btn_jump_frame_click
        self.container_keyframe_options.append(self.btn_keyframe_prev)

        # btn_keyframe_next
        self.btn_keyframe_next = ScreenButton("btn_keyframe_next", "Next", 22, 75)
        self.btn_keyframe_next.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_keyframe_next.position = (225, self.lbl_keyframe_title.get_bottom() + 10)
        self.btn_keyframe_next.click_callback = self.btn_jump_frame_click
        self.container_keyframe_options.append(self.btn_keyframe_next)

        # btn_keyframe_add
        self.btn_keyframe_add = ScreenButton("btn_keyframe_add", "Add", 22, 75)
        self.btn_keyframe_add.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_keyframe_add.position = (325, self.lbl_keyframe_title.get_bottom() + 10)
        self.btn_keyframe_add.click_callback = self.btn_keyframe_add_click
        self.container_keyframe_options.append(self.btn_keyframe_add)

        # btn_keyframe_del
        self.btn_keyframe_del = ScreenButton("btn_keyframe_del", "Del", 22, 75)
        self.btn_keyframe_del.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_keyframe_del.position = (325, self.lbl_keyframe_title.get_bottom() + 10)
        self.btn_keyframe_del.click_callback = self.btn_keyframe_del_click
        self.container_keyframe_options.append(self.btn_keyframe_del)

        # lbl_keyframe_prev
        self.lbl_keyframe_prev = ScreenLabel("lbl_keyframe_prev", "[0]", 21, 75)
        self.lbl_keyframe_prev.position = (self.btn_keyframe_prev.get_left(), self.btn_keyframe_prev.get_bottom() + 10)
        self.lbl_keyframe_prev.set_color((255, 255, 255))
        self.lbl_keyframe_prev.set_background(self.general_background)
        self.container_keyframe_options.append(self.lbl_keyframe_prev)

        # lbl_keyframe_next
        self.lbl_keyframe_next = ScreenLabel("lbl_keyframe_next", "[0]", 21, 75)
        self.lbl_keyframe_next.position = (self.btn_keyframe_next.get_left(), self.btn_keyframe_next.get_bottom() + 10)
        self.lbl_keyframe_next.set_color((255, 255, 255))
        self.lbl_keyframe_next.set_background(self.general_background)
        self.container_keyframe_options.append(self.lbl_keyframe_next)

        # lbl_keyframe_label
        self.lbl_keyframe_label = ScreenLabel("lbl_keyframe_label", "Seg. LBL: ", 21, 280, centered=0)
        self.lbl_keyframe_label.position = (self.btn_keyframe_invisible.get_left(),
                                            self.lbl_keyframe_next.get_bottom() + 20)
        self.lbl_keyframe_label.set_color((255, 255, 255))
        self.lbl_keyframe_label.set_background(self.general_background)
        self.container_keyframe_options.append(self.lbl_keyframe_label)

        # txt_keyframe_label
        # self.txt_keyframe_label = ScreenTextbox("txt_keyframe_label", "", 25, 230)
        # self.txt_keyframe_label.position = (self.lbl_keyframe_label.get_right() + 10,
        #                                     self.lbl_keyframe_next.get_bottom() + 15)
        # self.txt_keyframe_label.set_colors((255, 255, 255), (40, 40, 48))
        # self.container_keyframe_options.append(self.txt_keyframe_label)

        self.btn_keyframe_label_set = ScreenButton("btn_keyframe_label_set", "Set", 22, 75)
        self.btn_keyframe_label_set.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_keyframe_label_set.position = (325, self.lbl_keyframe_next.get_bottom() + 15)
        self.btn_keyframe_label_set.click_callback = self.btn_keyframe_label_set_click
        self.container_keyframe_options.append(self.btn_keyframe_label_set)

        self.elements.append(self.container_keyframe_options)

    def create_video_segments_buttons(self):
        self.container_vid_seg_options = ScreenContainer("container_vid_seg_options", (425, 100),
                                                         self.general_background)
        self.container_vid_seg_options.position = (self.width - self.container_vid_seg_options.width - 10,
                                                   self.container_keyframe_options.get_bottom() + 5)
        self.container_vid_seg_options.visible = True

        # lbl_vid_seg_title
        self.lbl_vid_seg_title = ScreenLabel("lbl_vid_seg_title", "Video Segments: ", 21, 415, 1)
        self.lbl_vid_seg_title.set_color((255, 255, 255))
        self.lbl_vid_seg_title.set_background(self.general_background)
        self.lbl_vid_seg_title.position = (5, 5)
        self.container_vid_seg_options.append(self.lbl_vid_seg_title)

        # btn_vid_seg_prev
        self.btn_vid_seg_prev = ScreenButton("btn_vid_seg_prev", "Prev", 22, 110)
        self.btn_vid_seg_prev.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_vid_seg_prev.position = (20, self.lbl_vid_seg_title.get_bottom() + 10)
        self.btn_vid_seg_prev.click_callback = self.btn_jump_frame_click
        self.container_vid_seg_options.append(self.btn_vid_seg_prev)

        # btn_vid_seg_next
        self.btn_vid_seg_next = ScreenButton("btn_vid_seg_next", "Next", 22, 110)
        self.btn_vid_seg_next.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_vid_seg_next.position = (157, self.lbl_vid_seg_title.get_bottom() + 10)
        self.btn_vid_seg_next.click_callback = self.btn_jump_frame_click
        self.container_vid_seg_options.append(self.btn_vid_seg_next)

        # btn_vid_seg_split
        self.btn_vid_seg_split = ScreenButton("btn_vid_seg_split", "Split", 22, 110)
        self.btn_vid_seg_split.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_vid_seg_split.position = (295, self.lbl_vid_seg_title.get_bottom() + 10)
        self.btn_vid_seg_split.click_callback = self.btn_vid_seg_split_click
        self.container_vid_seg_options.append(self.btn_vid_seg_split)

        # btn_vid_seg_merge
        self.btn_vid_seg_merge = ScreenButton("btn_vid_seg_merge", "Merge", 22, 110)
        self.btn_vid_seg_merge.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_vid_seg_merge.position = (295, self.lbl_vid_seg_title.get_bottom() + 10)
        self.btn_vid_seg_merge.click_callback = self.btn_vid_seg_merge_click
        self.container_vid_seg_options.append(self.btn_vid_seg_merge)

        # lbl_vid_seg_prev
        self.lbl_vid_seg_prev = ScreenLabel("lbl_vid_seg_prev", "[0]", 21, 110)
        self.lbl_vid_seg_prev.position = (self.btn_vid_seg_prev.get_left(), self.btn_vid_seg_prev.get_bottom() + 10)
        self.lbl_vid_seg_prev.set_color((255, 255, 255))
        self.lbl_vid_seg_prev.set_background(self.general_background)
        self.container_vid_seg_options.append(self.lbl_vid_seg_prev)

        # lbl_vid_seg_next
        self.lbl_vid_seg_next = ScreenLabel("lbl_vid_seg_next", "[0]", 21, 110)
        self.lbl_vid_seg_next.position = (self.btn_vid_seg_next.get_left(), self.btn_vid_seg_next.get_bottom() + 10)
        self.lbl_vid_seg_next.set_color((255, 255, 255))
        self.lbl_vid_seg_next.set_background(self.general_background)
        self.container_vid_seg_options.append(self.lbl_vid_seg_next)

        self.elements.append(self.container_vid_seg_options)

    def create_video_segments_keyframes_buttons(self):
        self.container_vid_seg_keyframe_options = ScreenContainer("container_vid_seg_keyframe_options", (425, 100),
                                                                  self.general_background)
        self.container_vid_seg_keyframe_options.position = (self.width - self.container_vid_seg_keyframe_options.width - 20,
                                                            self.container_vid_seg_options.get_bottom() + 5)
        self.container_vid_seg_keyframe_options.visible = True

        # lbl_vid_seg_keyframe_title
        self.lbl_vid_seg_keyframe_title = ScreenLabel("lbl_vid_seg_keyframe_title", "Segment Keyframes: ", 21, 415, 1)
        self.lbl_vid_seg_keyframe_title.set_color((255, 255, 255))
        self.lbl_vid_seg_keyframe_title.set_background(self.general_background)
        self.lbl_vid_seg_keyframe_title.position = (5, 5)
        self.container_vid_seg_keyframe_options.append(self.lbl_vid_seg_keyframe_title)

        # btn_vid_seg_keyframe_prev
        self.btn_vid_seg_keyframe_prev = ScreenButton("btn_vid_seg_keyframe_prev", "Prev", 22, 110)
        self.btn_vid_seg_keyframe_prev.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_vid_seg_keyframe_prev.position = (20, self.lbl_vid_seg_keyframe_title.get_bottom() + 10)
        self.btn_vid_seg_keyframe_prev.click_callback = self.btn_jump_frame_click
        self.container_vid_seg_keyframe_options.append(self.btn_vid_seg_keyframe_prev)

        # btn_vid_seg_keyframe_next
        self.btn_vid_seg_keyframe_next = ScreenButton("btn_vid_seg_keyframe_next", "Next", 22, 110)
        self.btn_vid_seg_keyframe_next.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_vid_seg_keyframe_next.position = (157, self.lbl_vid_seg_keyframe_title.get_bottom() + 10)
        self.btn_vid_seg_keyframe_next.click_callback = self.btn_jump_frame_click
        self.container_vid_seg_keyframe_options.append(self.btn_vid_seg_keyframe_next)

        # btn_vid_seg_keyframe_add
        self.btn_vid_seg_keyframe_add = ScreenButton("btn_vid_seg_keyframe_split", "Add", 22, 110)
        self.btn_vid_seg_keyframe_add.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_vid_seg_keyframe_add.position = (295, self.lbl_vid_seg_keyframe_title.get_bottom() + 10)
        self.btn_vid_seg_keyframe_add.click_callback = self.btn_vid_seg_keyframe_add_click
        self.container_vid_seg_keyframe_options.append(self.btn_vid_seg_keyframe_add)

        # btn_vid_seg_keyframe_del
        self.btn_vid_seg_keyframe_del = ScreenButton("btn_vid_seg_keyframe_del", "Del", 22, 110)
        self.btn_vid_seg_keyframe_del.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_vid_seg_keyframe_del.position = (295, self.lbl_vid_seg_keyframe_title.get_bottom() + 10)
        self.btn_vid_seg_keyframe_del.click_callback = self.btn_vid_seg_keyframe_del_click
        self.container_vid_seg_keyframe_options.append(self.btn_vid_seg_keyframe_del)

        # lbl_vid_seg_keyframe_prev
        self.lbl_vid_seg_keyframe_prev = ScreenLabel("lbl_vid_seg_keyframe_prev", "[0]", 21, 110)
        self.lbl_vid_seg_keyframe_prev.position = (self.btn_vid_seg_keyframe_prev.get_left(),
                                                   self.btn_vid_seg_keyframe_prev.get_bottom() + 10)
        self.lbl_vid_seg_keyframe_prev.set_color((255, 255, 255))
        self.lbl_vid_seg_keyframe_prev.set_background(self.general_background)
        self.container_vid_seg_keyframe_options.append(self.lbl_vid_seg_keyframe_prev)

        # lbl_vid_seg_keyframe_next
        self.lbl_vid_seg_keyframe_next = ScreenLabel("lbl_vid_seg_keyframe_next", "[0]", 21, 110)
        self.lbl_vid_seg_keyframe_next.position = (self.btn_vid_seg_keyframe_next.get_left(),
                                                   self.btn_vid_seg_keyframe_next.get_bottom() + 10)
        self.lbl_vid_seg_keyframe_next.set_color((255, 255, 255))
        self.lbl_vid_seg_keyframe_next.set_background(self.general_background)
        self.container_vid_seg_keyframe_options.append(self.lbl_vid_seg_keyframe_next)

        self.elements.append(self.container_vid_seg_keyframe_options)

    def create_text_input_buttons(self):
        # Container for text input ....
        self.container_text_input = ScreenContainer("container_text_input", (425, 230), self.general_background)
        self.container_text_input.position = (self.width - self.container_text_input.width - 20, self.canvas.get_top())

        # ...text box ...
        self.txt_object_name = ScreenTextbox("txt_object_name", "", 25, 280)
        self.txt_object_name.position = (int((self.container_text_input.width - self.txt_object_name.width) / 2), 80)
        self.txt_object_name.set_colors((255, 255, 255), (40, 40, 48))
        self.container_text_input.append(self.txt_object_name)
        # ...title for text box ...
        self.label_txt_object_name = ScreenLabel("label_txt_object_name", "Object Name", 21, max_width=280)
        self.label_txt_object_name.position = (self.txt_object_name.get_left(), 60)
        self.label_txt_object_name.set_color((255, 255, 255))
        self.label_txt_object_name.set_background(self.general_background)
        self.container_text_input.append(self.label_txt_object_name)
        # ... accept button ...
        self.btn_text_operation_accept = ScreenButton("btn_text_operation_accept", "Accept", 22, 100)
        self.btn_text_operation_accept.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_text_operation_accept.position = (self.txt_object_name.get_center_x() - 120,
                                                   self.txt_object_name.get_bottom() + 30)
        self.btn_text_operation_accept.click_callback = self.btn_text_operation_accept_click
        self.container_text_input.append(self.btn_text_operation_accept)
        # ... cancel button ...
        self.btn_text_operation_cancel = ScreenButton("btn_text_operation_cancel", "Cancel", 22, 100)
        self.btn_text_operation_cancel.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_text_operation_cancel.position = (self.txt_object_name.get_center_x() + 20,
                                                   self.txt_object_name.get_bottom() + 30)
        self.btn_text_operation_cancel.click_callback = self.btn_text_operation_cancel_click
        self.container_text_input.append(self.btn_text_operation_cancel)
        # ...error display label...
        self.label_txt_object_error = ScreenLabel("label_txt_object_error", "Displaying Error Messages", 21,
                                                  max_width=280)
        self.label_txt_object_error.position = (self.txt_object_name.get_left(),
                                                self.btn_text_operation_accept.get_bottom() + 15)
        self.label_txt_object_error.set_color((255, 64, 64))
        self.label_txt_object_error.set_background(self.general_background)
        self.container_text_input.append(self.label_txt_object_error)

        self.container_text_input.visible = False
        self.elements.append(self.container_text_input)

    def create_segment_labels_buttons(self):
        # =============================================
        # special container for buttons with options for quick labeling
        self.container_segment_labels = ScreenContainer("container_segment_labels", (425, 280), self.general_background)
        self.container_segment_labels.position = (self.width - self.container_segment_labels.width - 20,
                                                  self.container_text_input.get_bottom() + 10)
        self.container_segment_labels.visible = False
        self.elements.append(self.container_segment_labels)

        for btn_idx in range(12):
            btn_prev_segment = ScreenButton("btn_prev_segment_labels_" + str(btn_idx), "[NONE]", 16, 200, 0)
            btn_prev_segment.set_colors((192, 255, 128), (64, 64, 64))

            left_prc = (0.25 + 0.5 * (btn_idx % 2))
            left = int(self.container_segment_labels.width * left_prc - btn_prev_segment.width * 0.50)

            btn_prev_segment.position = (left, 15 + 45 * int(btn_idx / 2))
            btn_prev_segment.tag = None
            btn_prev_segment.click_callback = self.btn_prev_segment_click
            self.container_segment_labels.append(btn_prev_segment)
            self.buttons_prev_segment_labels.append(btn_prev_segment)

    def create_general_buttons(self):
        # SAVE BUTTON
        self.save_button = ScreenButton("save_button", "SAVE", 16, 100, 0)
        self.save_button.set_colors((192, 255, 128), (64, 64, 64))
        self.save_button.position = (self.exit_button.get_left() - self.save_button.width - 20,
                                     self.height - self.save_button.height - 15)
        self.save_button.click_callback = self.save_data_click
        self.elements.append(self.save_button)

        # EXPORT Segments Button
        self.export_button = ScreenButton("export_button", "EXPORT SEGMENTS", 16, 150, 0)
        self.export_button.set_colors((192, 255, 128), (64, 64, 64))
        self.export_button.position = (self.save_button.get_left() - self.export_button.width - 20,
                                       self.height - self.export_button.height - 15)
        self.export_button.click_callback = self.btn_export_segments_click
        self.elements.append(self.export_button)

        # EXPORT Object Button
        self.export_object_button = ScreenButton("export_object_button", "EXPORT OBJECT", 16, 150, 0)
        self.export_object_button.set_colors((192, 255, 128), (64, 64, 64))
        self.export_object_button.position = (self.export_button.get_left() - self.export_object_button.width - 20,
                                              self.height - self.export_object_button.height - 15)
        self.export_object_button.click_callback = self.btn_export_object_click
        self.elements.append(self.export_object_button)

        # REDO Button
        self.redo_button = ScreenButton("redo_button", "REDO", 16, 100, 0)
        self.redo_button.set_colors((192, 255, 128), (64, 64, 64))
        self.redo_button.position = (self.export_object_button.get_left() - self.redo_button.width - 50,
                                     self.height - self.redo_button.height - 15)
        self.redo_button.click_callback = self.btn_redo_click
        self.elements.append(self.redo_button)

        # UNDO Button
        self.undo_button = ScreenButton("undo_button", "UNDO", 16, 100, 0)
        self.undo_button.set_colors((192, 255, 128), (64, 64, 64))
        self.undo_button.position = (self.redo_button.get_left() - self.redo_button.width - 20,
                                     self.height - self.undo_button.height - 15)
        self.undo_button.click_callback = self.btn_undo_click
        self.elements.append(self.undo_button)

    def forced_exit_clicked(self, event):
        self.close_click(self.exit_button)

    def auto_save_timer_tick(self, timer):
        if not self.changes_saved:
            # auto-save .... to back up file
            xml_data = self.lecture.generate_data_xml()

            out_file = open(self.output_filename + ".bak", "w")
            out_file.write(xml_data)
            out_file.close()

            print("Auto-Backed up to: " + self.output_filename + ".bak")

    def load_saved_data_into_GUI(self):
        # load video objects
        for video_object_id in self.lecture.video_objects:
            # load logical object ...
            video_object = self.lecture[video_object_id]

            # add to the interface
            # ... txtlist
            self.object_selector.add_option(video_object.id, video_object.name)

            # ... canvas
            if video_object.is_rectangle():
                self.canvas.add_rectangle_element(video_object.id, 0, 0, 0, 0)
            else:
                n_points = video_object.polygon_points()
                zero_loc = np.zeros((n_points, 2), dtype=np.float64)
                self.canvas.add_polygon_element(video_object.id, zero_loc)

            self.canvas.elements[video_object.id].visible = False


    def btn_undo_click(self, button):
        if len(self.undo_stack) == 0:
            print("No operations to undo")
            return

        # copy last operation
        to_undo = self.undo_stack[-1]

        success = False
        if to_undo["operation"] == "object_added":
            # inverse of adding is removing ...
            success = self.remove_object(to_undo["id"])

        elif to_undo["operation"] == "object_renamed":
            # inverse of renaming, is going back to old name  ...
            success = self.rename_object(to_undo["new_id"], to_undo["old_id"], to_undo["old_display"])

        elif to_undo["operation"] == "object_removed":
            # inverse of removing, adding back ..
            old_object = to_undo["object_ref"]
            first_loc = old_object.locations[0]
            success = self.add_object(to_undo["id"], to_undo["display"], 0, 0, old_object.shape_type,
                                      first_loc.polygon_points)
            if success:
                new_object = self.lecture[to_undo["id"]]
                # overwrite locations
                new_object.locations = old_object.locations
                # update object reference
                to_undo["object_ref"] = new_object

        elif to_undo["operation"] == "keyframe_added":
            success = self.lecture[to_undo["object_id"]].del_location_at(to_undo["new_location"].frame)

        elif to_undo["operation"] == "keyframe_edited" or to_undo["operation"] == "keyframe_deleted":
            # return key-frame to previous state (either modify or add back)
            pre_loc = to_undo["old_location"]
            self.lecture[to_undo["object_id"]].set_location_at(pre_loc.frame, pre_loc.abs_time,
                                                                             pre_loc.visible, pre_loc.polygon_points)
            success = True

        elif to_undo["operation"] == "vid_seg_split":
            # to undo split ... merge
            success = self.segment_merge(to_undo["split_point"], False)
        elif to_undo["operation"] == "vid_seg_merge":
            # to undo merge ... split
            success = self.segment_split(to_undo["split_point"], False)
        elif to_undo["operation"] == "vid_seg_keyframe_add":
            # del key-frame
            success = self.segment_keyframe_del(to_undo["frame_index"], False)
        elif to_undo["operation"] == "vid_seg_keyframe_del":
            # add key-frame
            success = self.segment_keyframe_add(to_undo["frame_index"], False)

        # removing ...
        if success:
            self.redo_stack.append(to_undo)
            del self.undo_stack[-1]

            # update canvas ...
            self.update_canvas_objects()

            # update key-frame information
            self.update_keyframe_buttons()
            self.update_video_segment_buttons()
        else:
            print("Action could not be undone")

    def btn_redo_click(self, button):
        if len(self.redo_stack) == 0:
            print("No operations to be re-done")
            return

        # copy last operation
        to_redo = self.redo_stack[-1]

        success = False
        if to_redo["operation"] == "object_added":
            loc = to_redo["location"]
            success = self.add_object(to_redo["id"], to_redo["name"], loc.frame, loc.abs_time, to_redo["shape"],
                                      loc.polygon_points)

        elif to_redo["operation"] == "object_renamed":
            success = self.rename_object(to_redo["old_id"], to_redo["new_id"], to_redo["new_display"])

        elif to_redo["operation"] == "object_removed":
            success = self.remove_object(to_redo["id"])

        elif to_redo["operation"] == "keyframe_added" or to_redo["operation"] == "keyframe_edited":
            add_loc = to_redo["new_location"]
            self.lecture[to_redo["object_id"]].set_location_at(add_loc.frame, add_loc.abs_time,
                                                                             add_loc.visible, add_loc.polygon_points)
            success = True

        elif to_redo["operation"] == "keyframe_deleted":
            # return key-frame to previous state (either modify or add back)
            success = self.lecture[to_redo["object_id"]].del_location_at(to_redo["old_location"].frame)

        elif to_redo["operation"] == "vid_seg_split":
            success = self.segment_split(to_redo["split_point"], False)
        elif to_redo["operation"] == "vid_seg_merge":
            success = self.segment_merge(to_redo["split_point"], False)
        elif to_redo["operation"] == "vid_seg_keyframe_add":
            success = self.segment_keyframe_add(to_redo["frame_index"], False)
        elif to_redo["operation"] == "vid_seg_keyframe_del":
            success = self.segment_keyframe_del(to_redo["frame_index"], False)

        if success:
            self.undo_stack.append(to_redo)
            # removing last operation
            del self.redo_stack[-1]

            # update canvas ...
            self.update_canvas_objects()

            # update key-frame information
            self.update_keyframe_buttons()
            self.update_video_segment_buttons()
        else:
            print("Action could not be re-done!")

    def add_object(self, id, name, frame, abs_time, shape_type, polygon_points):
        # add to objects ...
        if not self.lecture.add_object(id, name, shape_type, frame, abs_time, polygon_points):
            print("The Object named <" + id + "> already exists!")
            return False

        if self.player.video_player.zoom_factor > 0:
            translation, scale = self.compute_canvas_zoom_translation_scale()

            canvas_polygon = (polygon_points * scale) + translation
        else:
            canvas_polygon = polygon_points

        # add to canvas ....
        if shape_type == VideoObject.ShapeAlignedRectangle:
            x, y = canvas_polygon[0]
            # right bottom  - left top
            w, h = canvas_polygon[2] - canvas_polygon[0]
            self.canvas.add_rectangle_element(id, x, y, w, h)
        elif shape_type in [VideoObject.ShapeQuadrilateral, VideoObject.ShapePolygon]:
            self.canvas.add_polygon_element(id, canvas_polygon)

        # add to text list
        self.object_selector.add_option(id, name)

        self.changes_saved = False

        return True

    def rename_object(self, old_id, new_id, new_name):
        if old_id != new_id:
            # name changed!, verify ...
            if not self.lecture.rename_object(old_id, new_id, new_name):
                print("Object name already in use")
                return False

            # valid name change, call rename operations
            # ... canvas ...
            self.canvas.rename_element(old_id, new_id)
            # ... object selector ...
            self.object_selector.rename_option(old_id, new_id, new_name)

            self.changes_saved = False

        return True

    def remove_object(self, object_name):
        if not self.lecture.remove_object(object_name):
            print("Cannot remove object")
            return False

        # ... remove from canvas
        self.canvas.remove_element(object_name)
        # ... remove from object selector
        self.object_selector.remove_option(object_name)

        self.changes_saved = False

        return True

    def main_scroll_change(self, scroll):
        self.player.set_player_frame(int(scroll.value), True)

    def compute_canvas_zoom_translation_scale(self):
        # find origin point
        viewport_scale = self.player.render_width / self.player.video_player.width
        o_x = self.player.video_player.visible_left() * viewport_scale
        o_y = self.player.video_player.visible_top() * viewport_scale

        # ... find gap ...
        gap_x, gap_y = self.lecture.drawing_info.canvas_render_dist()

        scale = pow(2, self.player.video_player.zoom_factor)

        t_x = gap_x - scale * (gap_x + o_x)
        t_y = gap_y - scale * (gap_y + o_y)

        translation = np.array([t_x, t_y])

        return translation, scale

    def update_canvas_objects(self):
        # compute visual tranformation ...

        # Linear transformation of the form
        # x_n = s_x x_o + t_x
        # y_n = s_y y_o + t_y
        # (first scale, then translate)

        if self.player.video_player.zoom_factor == 0:
            translation = None
            scale = None
        else:
            translation, scale = self.compute_canvas_zoom_translation_scale()

        # update canvas objects ....
        for object_name in self.lecture.video_objects:
            shape = self.lecture[object_name].shape_type

            if object_name == self.object_selector.selected_option_value:
                # selected object .... force an out of range location
                loc = self.lecture[object_name].get_location_at(self.last_video_frame, True)
            else:
                # any other object, no out of range location given
                loc = self.lecture[object_name].get_location_at(self.last_video_frame, False)

            # check ....
            if loc is None:
                # out of range ...
                if shape == VideoObject.ShapeAlignedRectangle:
                    self.canvas.update_rectangle_element(object_name, 0, 0, 0, 0, False, 1)
                elif shape in [VideoObject.ShapeQuadrilateral, VideoObject.ShapePolygon]:
                    self.canvas.update_polygon_element(object_name, None, False, 1)
                else:
                    raise Exception("Unknown Video Object Shape")
            else:
                # check if out of range (only applies to selected object) ....
                selected_out = (self.last_video_frame < self.lecture[object_name].first_frame() or
                                self.lecture[object_name].last_frame() < self.last_video_frame)

                # selected object will be drawn using dashed lines if out of range ....
                n_dashes = 1 if not selected_out else 100

                if self.player.video_player.zoom_factor == 0:
                    # Default, no zoom
                    loc_points = loc.polygon_points
                else:
                    # scaled view ...
                    loc_points = (loc.polygon_points * scale) + translation

                # in range or selected object ... just draw normally ....
                if shape == VideoObject.ShapeAlignedRectangle:
                    x, y = loc_points[0]
                    w, h = loc_points[2] - loc_points[0]

                    self.canvas.update_rectangle_element(object_name, x, y, w, h, loc.visible, n_dashes)
                elif shape in [VideoObject.ShapeQuadrilateral, VideoObject.ShapePolygon]:

                    self.canvas.update_polygon_element(object_name, loc_points, loc.visible, n_dashes)
                else:
                    raise Exception("Unknown Video Object Shape")


    def video_frame_change(self, next_frame, next_abs_time):
        # update the scroll bar
        self.position_scroll.value = next_frame

        self.last_video_frame = next_frame
        self.last_video_time = next_abs_time

        self.label_frame_current.set_text("Current frame: " + str(next_frame))
        self.label_time_current.set_text("Current time: " + TimeHelper.stampToStr(next_abs_time))

        # update canvas ...
        self.update_canvas_objects()

        # update key-frame information
        self.update_keyframe_buttons()

        # udpate segment info ...
        self.update_video_segment_buttons()


    def handle_events(self, events):
        #handle other events
        return super(GTContentAnnotator, self).handle_events(events)


    def render(self, background):
        # draw other controls..
        super(GTContentAnnotator, self).render(background)

    def close_click(self, button):
        if self.changes_saved:
            self.return_screen = None
            print("APPLICATION FINISHED")
        else:
            self.text_operation = 3
            print("Warning: Last changes have not been saved")
            # set exit confirm mode
            self.prepare_confirm_input_mode(4, None, "Exit without saving?")

    def save_data_click(self, button):
        xml_data = self.lecture.generate_data_xml()

        out_file = open(self.output_filename, "w")
        out_file.write(xml_data)
        out_file.close()

        print("Saved to: " + self.output_filename)
        self.changes_saved = True
        # Free the queues
        self.undo_stack.clear()
        self.redo_stack.clear()

    def btn_dec_speed_click(self, button):
        self.player.decrease_speed()

        self.label_player_speed.set_text("Speed: " + str(self.player.video_player.play_speed * 100.0) + "%")

    def btn_inc_speed_click(self, button):
        self.player.increase_speed()

        self.label_player_speed.set_text("Speed: " + str(self.player.video_player.play_speed * 100.0) + "%")

    def btn_change_frame(self, button):
        new_abs_frame = self.player.video_player.last_frame_idx + button.tag
        self.player.set_player_frame(new_abs_frame, True)

    def btn_pause_click(self, button):
        self.player.pause()
        self.canvas.locked = False

        self.button_play.visible = True
        self.button_pause.visible = False

    def btn_play_click(self, button):
        self.player.play()
        self.canvas.locked = True

        self.button_play.visible = False
        self.button_pause.visible = True

    def canvas_object_edited(self, canvas, object_name):
        #print("Object <" + str(object_name) + "> was edited")
        if self.lecture[object_name].shape_type == VideoObject.ShapeAlignedRectangle:
            # a rectangle ... convert to array of points (polygon format)
            x = canvas.elements[object_name].x
            y = canvas.elements[object_name].y
            w = canvas.elements[object_name].w
            h = canvas.elements[object_name].h
            canvas_polygon = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)
        elif self.lecture[object_name].shape_type in [VideoObject.ShapeQuadrilateral, VideoObject.ShapePolygon]:
            # should be a polygon already
            canvas_polygon = canvas.elements[object_name].points
        else:
            raise Exception("Invalid VideoObject Shape Type")

        prev_location = self.lecture[object_name].get_location_at(self.last_video_frame, False)

        translation, scale = self.compute_canvas_zoom_translation_scale()
        if self.player.video_player.zoom_factor == 0:
            object_polygon = canvas_polygon
        else:
            object_polygon = (canvas_polygon - translation) / scale

        keyframe_added = self.lecture[object_name].set_location_at(self.last_video_frame, self.last_video_time, True,
                                                                   object_polygon)

        if keyframe_added:
            # Do not store interpolated locations
            prev_location = None
        else:
            # use a copy of the location provided
            prev_location = VideoObjectLocation.fromLocation(prev_location)

        self.changes_saved = False
        object_location = VideoObjectLocation(True, self.last_video_frame, self.last_video_time, object_polygon)

        # check if the object was the last object edited
        if keyframe_added:
            # new key-frame
            self.undo_stack.append({
                "operation": "keyframe_added",
                "object_id": object_name,
                "old_location": prev_location,
                "new_location": object_location,
            })

        else:
            # edited key-frame, check if same as last change
            if (len(self.undo_stack) > 0 and
                self.undo_stack[-1]["operation"] == "keyframe_edited" and
                self.undo_stack[-1]["object_id"] == object_name and
                self.undo_stack[-1]["new_location"].frame == self.last_video_frame and
                time.time() - self.undo_stack[-1]["time"] < GTContentAnnotator.EditGroupingTime):
                # same object was modified last within n seconds, combine
                self.undo_stack[-1]["new_location"] = object_location
            else:
                # first modification to this object will be added to the top of the stack
                self.undo_stack.append({
                    "operation": "keyframe_edited",
                    "object_id": object_name,
                    "old_location": prev_location,
                    "new_location": object_location,
                    "time": time.time()
                })

        self.update_keyframe_buttons()

    def btn_object_add_bbox_click(self, button):
        # set adding mode
        self.new_object_shape = VideoObject.ShapeAlignedRectangle
        self.prepare_confirm_input_mode(-1, "", None)

    def btn_object_add_quad_click(self, button):
        # set adding mode
        self.new_object_shape = VideoObject.ShapeQuadrilateral
        self.prepare_confirm_input_mode(-1, "", None)

    def btn_object_rename_click(self, button):
        # first, check an object is selected on the list
        selected_name = self.object_selector.selected_option_value
        if selected_name is None:
            # nothing is selected
            return

        # set rename mode
        self.prepare_confirm_input_mode(2, selected_name, None)

    def btn_object_remove_click(self, button):
        # first, check an object is selected on the list
        selected_name = self.object_selector.selected_option_value
        if selected_name is None:
            # nothing is selected
            return

        # set delete mode
        self.prepare_confirm_input_mode(3, None, "Are you sure?")

    def prepare_confirm_input_mode(self, text_operation, textbox_text, message_text):
        # first, pause the video (if playing)
        self.btn_pause_click(None)

        # Now, change containers
        self.canvas.locked = True
        self.container_object_options.visible = False
        self.container_video_controls.visible = False
        self.panning_ver_scroll.visible = False
        self.container_keyframe_options.visible = False
        self.container_text_input.visible = True
        self.container_segment_labels.visible = (text_operation == 5)

        # text input only visible for adding/renaming/labeling video objects
        if text_operation == 1 or text_operation == 2:
            self.label_txt_object_name.set_text("Object Name")
        elif text_operation == -1:
            self.label_txt_object_name.set_text("Click on Object Position")
        elif text_operation == 5:
            self.label_txt_object_name.set_text("Object - Label for Segment")

        self.txt_object_name.visible = (text_operation == 1 or text_operation == 2 or text_operation == 5)
        self.label_txt_object_name.visible = (text_operation == -1 or text_operation == 1 or
                                              text_operation == 2 or text_operation == 5)

        self.btn_text_operation_accept.visible = (text_operation != -1)
        if self.txt_object_name.visible:
            self.txt_object_name.set_focus()

        # Text ...
        if text_operation == 1 and self.user_last_name_prefix is not None:
            self.txt_object_name.updateText(self.user_last_name_prefix)
        elif textbox_text is not None:
            self.txt_object_name.updateText(textbox_text)

        # Message Text ...
        if message_text is None:
            self.label_txt_object_error.visible = False
        else:
            self.label_txt_object_error.set_text(message_text)
            self.label_txt_object_error.visible = True

        self.text_operation = text_operation

    def btn_text_operation_accept_click(self, button):
        new_name = self.txt_object_name.text.strip()
        id_name = new_name.lower()

        # name must not be empty
        if new_name == "" and (self.text_operation == 1 or self.text_operation == 2):
            self.label_txt_object_error.set_text("Object name cannot be empty")
            self.label_txt_object_error.visible = True
            return

        if self.text_operation == 1:
            # add, validate ...
            if id_name[-1] == "_":
                # the given name ends with underscore ..
                # find other objects with same prefix and determine next number in sequence
                next_int_corr = self.lecture.get_next_object_name_correlative(id_name)
                next_str_corr = str(next_int_corr).zfill(3)

                # store prefix to avoid typing it again next time
                self.user_last_name_prefix = new_name

                # update name
                id_name = id_name + next_str_corr
                new_name = new_name + next_str_corr
            else:
                # no prefix used ...
                self.user_last_name_prefix = None

            # ... check unique ...
            if self.lecture.contains(id_name):
                self.label_txt_object_error.set_text("Object name already in use")
                self.label_txt_object_error.visible = True
                return

            if (self.new_object_shape == VideoObject.ShapeAlignedRectangle or
                self.new_object_shape == VideoObject.ShapeQuadrilateral):
                obj_x, obj_y = self.user_selected_position
                canvas_position = np.array([[obj_x, obj_y], [obj_x + 100, obj_y],
                                             [obj_x + 100, obj_y + 50], [obj_x, obj_y + 50]], dtype=np.float64)

                if self.player.video_player.zoom_factor > 0:
                    translation, scale = self.compute_canvas_zoom_translation_scale()

                    default_position = (canvas_position - translation) / scale
                else:
                    default_position = canvas_position

            else:
                raise Exception("Invalid VideoObject Shape")

            # valid... add!
            if self.add_object(id_name, new_name, self.last_video_frame, self.last_video_time, self.new_object_shape, default_position):
                location = VideoObjectLocation(True, self.last_video_frame, self.last_video_time, default_position)
                self.undo_stack.append({
                    "operation": "object_added",
                    "id": id_name,
                    "name": new_name,
                    "shape": self.new_object_shape,
                    "location": location,
                })
            else:
                return

        if self.text_operation == 2:
            # rename
            selected_name = self.object_selector.selected_option_value

            if selected_name != id_name:
                # name changed!, verify ...
                if self.lecture.contains(id_name):
                    self.label_txt_object_error.set_text("Object name already in use")
                    self.label_txt_object_error.visible = True
                    return

            old_display = self.object_selector.option_display[selected_name]
            if self.rename_object(selected_name, id_name, new_name):
                self.undo_stack.append({
                    "operation": "object_renamed",
                    "old_id": selected_name,
                    "old_display": old_display,
                    "new_id": id_name,
                    "new_display": new_name,
                })
            else:
                return

        if self.text_operation == 3:
            # delete (confirmed)
            selected_name = self.object_selector.selected_option_value

            removed_object = self.lecture[selected_name]
            removed_display = self.object_selector.option_display[selected_name]
            if self.remove_object(selected_name):
                self.undo_stack.append({
                    "operation": "object_removed",
                    "id": selected_name,
                    "display": removed_display,
                    "object_ref": removed_object,
                })

        if self.text_operation == 4:
            self.return_screen = None
            print("APPLICATION FINISHED / CHANGES LOST")

        if self.text_operation == 5:
            selected_name = self.object_selector.selected_option_value
            current_object = self.lecture[selected_name]
            current_object_loc = current_object.get_location_at(self.last_video_frame, False, False)
            if new_name == "":
                current_object_loc.label = None
            else:
                current_object_loc.label = new_name

        self.container_object_options.visible = True
        self.container_video_controls.visible = True
        self.panning_ver_scroll.visible = self.panning_hor_scroll.visible
        self.container_keyframe_options.visible = (self.object_selector.selected_option_value is not None)
        self.container_text_input.visible = False
        self.container_segment_labels.visible = False
        self.canvas.locked = False
        self.elements.set_text_focus(None)

        self.update_keyframe_buttons()

    def btn_text_operation_cancel_click(self, button):
        self.text_operation = 0

        self.container_object_options.visible = True
        self.container_video_controls.visible = True
        self.panning_ver_scroll.visible = self.panning_hor_scroll.visible
        self.container_keyframe_options.visible = (self.object_selector.selected_option_value is not None)
        self.container_text_input.visible = False
        self.container_segment_labels.visible = False
        self.elements.set_text_focus(None)
        self.canvas.locked = False

    def object_selector_option_changed(self, new_value, old_value):
        self.select_object(new_value, 1)

    def canvas_selection_changed(self, object_selected):
        self.select_object(object_selected, 2)

    def select_object(self, new_object, source):
        if source != 1:
            # mark object selector ...
            self.object_selector.change_option_selected(new_object)

        if source != 2:
            # select object in canvas ...
            self.canvas.change_selected_element(new_object)

        # source #3 = video player using right click ...

        self.container_keyframe_options.visible = new_object is not None
        if new_object is None:
            self.btn_object_split_poly.visible = False

        self.update_canvas_objects()
        self.update_keyframe_buttons()


    def btn_keyframe_visible_click(self, button):
        self.set_object_keyframe_visible(True)
        self.btn_keyframe_visible.visible = False
        self.btn_keyframe_invisible.visible = True

    def btn_keyframe_invisible_click(self, button):
        self.set_object_keyframe_visible(False)
        self.btn_keyframe_visible.visible = True
        self.btn_keyframe_invisible.visible = False

    def set_object_keyframe_visible(self, is_visible):
        current_frame = self.last_video_frame
        selected_name = self.canvas.selected_element

        if selected_name is not None:
            current_object = self.lecture[selected_name]
            # next/previous ....
            loc_idx = current_object.find_location_idx(current_frame)

            if not loc_idx >= len(current_object.locations) and current_object.locations[loc_idx].frame == current_frame:
                current_loc = current_object.locations[loc_idx]
                # copy before changing
                old_location = VideoObjectLocation.fromLocation(current_loc)

                #change ...
                current_loc.visible = is_visible
                self.canvas.elements[selected_name].visible = is_visible

                self.changes_saved = False

                # add to undo stack
                self.undo_stack.append({
                    "operation": "keyframe_edited",
                    "object_id": selected_name,
                    "old_location": old_location,
                    "new_location": VideoObjectLocation.fromLocation(current_loc),
                    "time": time.time(),
                })

    def btn_jump_frame_click(self, button):
        self.player.set_player_frame(button.tag, True)

    def btn_keyframe_add_click(self, button):
        current_frame = self.last_video_frame
        current_time = self.last_video_time
        selected_name = self.canvas.selected_element

        if selected_name is not None:
            current_object = self.lecture[selected_name]

            loc_idx = current_object.find_location_idx(current_frame)

            if loc_idx >= len(current_object.locations):
                # out of boundaries, after last one
                base_loc = current_object.locations[-1]
            else:
                if current_object.locations[0].frame > current_frame:
                    # out of boundaries, before first one
                    base_loc = current_object.locations[0]
                else:
                    # only add if the current frame is not a keyframe already ...
                    if current_object.locations[loc_idx].frame != current_frame:
                        # not key-frame and not out of boundaries
                        base_loc = VideoObjectLocation.interpolate(current_object.locations[loc_idx - 1],
                                                                   current_object.locations[loc_idx], current_frame)
                    else:
                        # already a key-frame
                        return

            current_object.set_location_at(current_frame, current_time, base_loc.visible, base_loc.polygon_points)

            if self.player.video_player.zoom_factor == 0:
                canvas_polygon = base_loc.polygon_points
            else:
                translation, scale = self.compute_canvas_zoom_translation_scale()
                canvas_polygon = base_loc.polygon_points * scale + translation

            if current_object.shape_type == VideoObject.ShapeAlignedRectangle:
                x, y = canvas_polygon[0]
                w, h = canvas_polygon[2] - canvas_polygon[0]
                self.canvas.update_rectangle_element(selected_name, x, y, w, h, base_loc.visible)
            elif current_object.shape_type in [VideoObject.ShapeQuadrilateral, VideoObject.ShapePolygon]:
                self.canvas.update_polygon_element(selected_name, canvas_polygon, base_loc.visible)

            self.changes_saved = False
            self.undo_stack.append({
                "operation": "keyframe_added",
                "object_id": selected_name,
                "old_location": None,
                "new_location": VideoObjectLocation.fromLocation(base_loc),
            })

        self.update_keyframe_buttons()

    def copy_neighbor_keyframe_location(self, copy_from_next):
        current_frame = self.last_video_frame
        current_time = self.last_video_time
        selected_name = self.canvas.selected_element

        if selected_name is not None:
            current_object = self.lecture[selected_name]

            # returns the "insert position" (where do I need to insert the new key-frame on this list)
            loc_idx = current_object.find_location_idx(current_frame)
            prev_loc = None

            if copy_from_next:
                # check if there is a next key-frame ...
                if loc_idx >= len(current_object.locations):
                    # there is no "next" frame to copy from ...
                    return
                elif current_frame < current_object.locations[0].frame:
                    # before first frame ...
                    base_loc = current_object.locations[0]
                elif current_object.locations[loc_idx].frame != current_frame:
                    # not a keyframe ... copy current loc which represents next kf
                    base_loc = current_object.locations[loc_idx]
                else:
                    # this location is already a key-frame
                    if loc_idx + 1 >= len(current_object.locations):
                        # there is no "next" frame to copy from ...
                        return
                    else:
                        base_loc = current_object.locations[loc_idx + 1]
                        # also copy current location before overwriting it
                        prev_loc = VideoObjectLocation.fromLocation(current_object.locations[loc_idx])
            else:
                # check if there is a previous key-frame
                if current_frame < current_object.locations[0].frame:
                    # before first frame ...
                    return
                elif current_object.locations[loc_idx].frame != current_frame:
                    # not a keyframe ... copy current loc - 1 which represents prev kf
                    base_loc = current_object.locations[loc_idx - 1]
                else:
                    if loc_idx == 0:
                        # we are exactly at the first keyframe and we cannot copy from anyone
                        return
                    else:
                        base_loc = current_object.locations[loc_idx - 1]
                        # also copy current location before overwriting it
                        prev_loc = VideoObjectLocation.fromLocation(current_object.locations[loc_idx])

            # if we reach this point is because there is a location to copy from ...
            current_object.set_location_at(current_frame, current_time, base_loc.visible, base_loc.polygon_points)

            if self.player.video_player.zoom_factor == 0:
                canvas_polygon = base_loc.polygon_points
            else:
                translation, scale = self.compute_canvas_zoom_translation_scale()
                canvas_polygon = base_loc.polygon_points * scale + translation

            if current_object.shape_type == VideoObject.ShapeAlignedRectangle:
                x, y = canvas_polygon[0]
                w, h = canvas_polygon[2] - canvas_polygon[0]
                self.canvas.update_rectangle_element(selected_name, x, y, w, h, base_loc.visible)
            elif current_object.shape_type in [VideoObject.ShapeQuadrilateral, VideoObject.ShapePolygon]:
                self.canvas.update_polygon_element(selected_name, canvas_polygon, base_loc.visible)

            self.changes_saved = False
            self.undo_stack.append({
                "operation": "keyframe_added",
                "object_id": selected_name,
                "old_location": prev_loc,
                "new_location": VideoObjectLocation.fromLocation(base_loc),
            })

        self.update_keyframe_buttons()

    def btn_keyframe_del_click(self, button):
        current_frame = self.last_video_frame
        selected_name = self.canvas.selected_element

        if selected_name is not None:
            current_object = self.lecture[selected_name]
            loc_idx = current_object.find_location_idx(current_frame)

            if (loc_idx < len(current_object.locations) and current_object.locations[loc_idx].frame == current_frame
                and len(current_object.locations) > 1):
                # a key-frame is selected, and is not the only one
                to_delete = current_object.locations[loc_idx]
                del current_object.locations[loc_idx]

                self.changes_saved = False
                self.undo_stack.append({
                    "operation": "keyframe_deleted",
                    "object_id": selected_name,
                    "old_location": VideoObjectLocation.fromLocation(to_delete),
                })

                # update everything ...
                # ... canvas ...
                self.update_canvas_objects()

                # ... key-frame buttons ...
                self.update_keyframe_buttons()

    def update_keyframe_buttons(self):
        current_frame = self.last_video_frame

        selected_name = self.canvas.selected_element

        kf_lbl_msg = "[{0:d}]-{1:d}"

        if selected_name is None:
            # count ...
            self.lbl_keyframe_title.set_text("Object Key-frames: [0]")

            # next/previous ....
            self.lbl_keyframe_prev.set_text("[0]-0")
            self.lbl_keyframe_next.set_text("[0]-0")
            # hide everything ...
            self.container_keyframe_options.visible = False
        else:
            current_object = self.lecture[selected_name]
            # count ...
            label_msg = "[{0:s}] Key-frames: {1:d}".format(selected_name, len(current_object.locations))
            self.lbl_keyframe_title.set_text(label_msg)

            # next/previous ....
            loc_idx = current_object.find_location_idx(current_frame)

            # make invisible by default
            self.btn_keyframe_visible.visible = False
            self.btn_keyframe_invisible.visible = False

            self.lbl_keyframe_prev.visible = False
            self.btn_keyframe_prev.visible = False
            self.btn_keyframe_prev.tag = None
            self.lbl_keyframe_next.visible = False
            self.btn_keyframe_next.visible = False
            self.btn_keyframe_next.tag = None
            self.lbl_keyframe_label.visible = False

            self.btn_keyframe_add.visible = True
            self.btn_keyframe_del.visible = False

            # the shape can be split if it is a quadrilateral or already a polygon
            self.btn_object_split_poly.visible = current_object.shape_type in [VideoObject.ShapePolygon,
                                                                               VideoObject.ShapeQuadrilateral]

            if loc_idx >= len(current_object.locations):
                # out of boundaries, next is none and prev is last
                self.lbl_keyframe_prev.visible = True
                self.btn_keyframe_prev.visible = True
                self.lbl_keyframe_prev.set_text(kf_lbl_msg.format(len(current_object.locations), current_object.locations[-1].frame))
                self.btn_keyframe_prev.tag = current_object.locations[-1].frame
                self.lbl_keyframe_next.set_text("[0]-0")

            else:
                if current_object.locations[0].frame > current_frame:
                    # out of boundaries, next is first and prev is None
                    self.lbl_keyframe_prev.set_text("[0]-0")
                    self.lbl_keyframe_next.visible = True
                    self.btn_keyframe_next.visible = True
                    self.lbl_keyframe_next.set_text(kf_lbl_msg.format(1, current_object.locations[0].frame))
                    self.btn_keyframe_next.tag = current_object.locations[0].frame
                else:
                    self.lbl_keyframe_label.visible = True

                    if current_object.locations[loc_idx].frame == current_frame:
                        self.btn_keyframe_add.visible = False
                        self.btn_keyframe_del.visible = len(current_object.locations) > 1

                        # on a key-frame
                        if loc_idx == 0:
                            # no previous ...
                            self.lbl_keyframe_prev.set_text("[X]-0")
                        else:
                            # previous keyframe is before current frame (which is a keyframe)
                            self.lbl_keyframe_prev.visible = True
                            self.btn_keyframe_prev.visible = True
                            self.lbl_keyframe_prev.set_text(kf_lbl_msg.format(loc_idx, current_object.locations[loc_idx - 1].frame))
                            self.btn_keyframe_prev.tag = current_object.locations[loc_idx - 1].frame

                        if loc_idx == len(current_object.locations) - 1:
                            # no next
                            self.lbl_keyframe_next.set_text("[X]-0")
                        else:
                            # next keyframe is after current frame (which is a keyframe)
                            self.lbl_keyframe_next.visible = True
                            self.btn_keyframe_next.visible = True
                            self.lbl_keyframe_next.set_text(kf_lbl_msg.format(loc_idx + 2, current_object.locations[loc_idx + 1].frame))
                            self.btn_keyframe_next.tag = current_object.locations[loc_idx + 1].frame

                        # update segment label
                        current_label = current_object.locations[loc_idx].label
                        label_txt = current_label if current_label is not None else ""
                        self.lbl_keyframe_label.set_text("Seg. LBL: " + label_txt)

                        # show the corresponding show/hide button ...
                        self.btn_keyframe_invisible.visible = current_object.locations[loc_idx].visible
                        self.btn_keyframe_visible.visible = not current_object.locations[loc_idx].visible
                    else:
                        # not key-frame and not out of boundaries

                        # previous keyframe is closest keyframe
                        self.lbl_keyframe_prev.visible = True
                        self.btn_keyframe_prev.visible = True
                        self.lbl_keyframe_prev.set_text(kf_lbl_msg.format(loc_idx, current_object.locations[loc_idx - 1].frame))
                        self.btn_keyframe_prev.tag = current_object.locations[loc_idx - 1].frame

                        # update segment label
                        current_label = current_object.locations[loc_idx - 1].label
                        label_txt = current_label if current_label is not None else ""
                        self.lbl_keyframe_label.set_text("Seg. LBL: " + label_txt)

                        # next keyframe is after current frame (which is a keyframe)
                        self.lbl_keyframe_next.visible = True
                        self.btn_keyframe_next.visible = True
                        self.lbl_keyframe_next.set_text(kf_lbl_msg.format(loc_idx + 1, current_object.locations[loc_idx].frame))
                        self.btn_keyframe_next.tag = current_object.locations[loc_idx].frame

    def update_video_segment_buttons(self):
        self.container_vid_seg_options.visible = self.last_video_frame is not None
        self.container_vid_seg_keyframe_options.visible = self.last_video_frame is not None
        if self.last_video_frame is None:
            return

        position = 0
        while position < len(self.lecture.video_segments) and self.last_video_frame > self.lecture.video_segments[position]:
            position += 1

        if position == len(self.lecture.video_segments):
            # at the end ...
            # check if unique segment too..
            if len(self.lecture.video_segments) > 0:
                prev_split = self.lecture.video_segments[-1]
            else:
                # unique element
                prev_split = 0
            next_split = None

            interval_start = prev_split
            interval_end = self.player.video_player.total_frames
        elif self.lecture.video_segments[position] == self.last_video_frame:
            # the exact element
            if position > 0:
                prev_split = self.lecture.video_segments[position - 1]
            else:
                prev_split = 0

            interval_start = self.lecture.video_segments[position]
            if position + 1 < len(self.lecture.video_segments):
                next_split = self.lecture.video_segments[position + 1]
                interval_end = next_split
            else:
                next_split = None
                interval_end = self.player.video_player.total_frames
        else:
            # other elements ...
            # check if beginning
            if position > 0:
                prev_split = self.lecture.video_segments[position - 1]
            else:
                prev_split = 0
            next_split = self.lecture.video_segments[position]

            interval_start = prev_split
            interval_end = next_split

        # Update the count
        txt_segments = "Video Segments: {0:d} / {1:d}"
        self.lbl_vid_seg_title.set_text(txt_segments.format(position + 1, len(self.lecture.video_segments) + 1))

        # prev/next segment
        self.btn_vid_seg_prev.visible = prev_split is not None
        self.lbl_vid_seg_prev.visible = prev_split is not None
        if prev_split is not None:
            self.btn_vid_seg_prev.tag = prev_split
            self.lbl_vid_seg_prev.set_text("[" + str(prev_split) + "]")

        self.btn_vid_seg_next.visible = next_split is not None
        self.lbl_vid_seg_next.visible = next_split is not None
        if next_split is not None:
            self.btn_vid_seg_next.tag = next_split
            self.lbl_vid_seg_next.set_text("[" + str(next_split) + "]")

        # segment split/merge
        not_first_or_last = 0 < self.last_video_frame < self.player.video_player.total_frames - 1
        self.btn_vid_seg_merge.visible = not_first_or_last and self.last_video_frame in self.lecture.video_segments
        self.btn_vid_seg_split.visible = not_first_or_last and self.last_video_frame not in self.lecture.video_segments

        # Determine the count of key-frames in the current segment
        interval_keyframes = [idx for idx in self.lecture.video_segment_keyframes
                              if interval_start <= idx < interval_end]

        position = 0
        while (position < len(self.lecture.video_segment_keyframes) and
               self.last_video_frame > self.lecture.video_segment_keyframes[position]):
            position += 1

        if position == len(self.lecture.video_segment_keyframes):
            # at the end ...
            # check if unique segment too..
            if len(self.lecture.video_segment_keyframes) > 0:
                prev_keyframe = self.lecture.video_segment_keyframes[-1]
            else:
                # unique element
                prev_keyframe = None
            next_keyframe = None

        elif self.lecture.video_segment_keyframes[position] == self.last_video_frame:
            # the exact element
            if position > 0:
                prev_keyframe = self.lecture.video_segment_keyframes[position - 1]
            else:
                prev_keyframe = None

            if position + 1 < len(self.lecture.video_segment_keyframes):
                next_keyframe = self.lecture.video_segment_keyframes[position + 1]
            else:
                next_keyframe = None
        else:
            # other elements ...
            # check if beginning
            if position > 0:
                prev_keyframe = self.lecture.video_segment_keyframes[position - 1]
            else:
                prev_keyframe = None
            next_keyframe = self.lecture.video_segment_keyframes[position]

        # prev/next segment
        self.btn_vid_seg_keyframe_prev.visible = prev_keyframe is not None
        self.lbl_vid_seg_keyframe_prev.visible = prev_keyframe is not None
        if prev_keyframe is not None:
            self.btn_vid_seg_keyframe_prev.tag = prev_keyframe
            self.lbl_vid_seg_keyframe_prev.set_text("[" + str(prev_keyframe) + "]")

        self.btn_vid_seg_keyframe_next.visible = next_keyframe is not None
        self.lbl_vid_seg_keyframe_next.visible = next_keyframe is not None
        if next_keyframe is not None:
            self.btn_vid_seg_keyframe_next.tag = next_keyframe
            self.lbl_vid_seg_keyframe_next.set_text("[" + str(next_keyframe) + "]")

        # Keyframe add/del
        self.btn_vid_seg_keyframe_del.visible = self.last_video_frame in self.lecture.video_segment_keyframes
        self.btn_vid_seg_keyframe_add.visible = self.last_video_frame not in self.lecture.video_segment_keyframes

        txt_vid_seg_kf = "Interval [{0:d}, {1:d}) Key-frames: {2:d} / {3:d}"
        txt_vid_seg_kf = txt_vid_seg_kf.format(interval_start, interval_end, len(interval_keyframes),
                                               len(self.lecture.video_segment_keyframes))
        self.lbl_vid_seg_keyframe_title.set_text(txt_vid_seg_kf)

    def segment_split(self, split_point, add_undo):
        first_or_last = 0 == split_point or split_point >= self.player.video_player.total_frames - 1

        if first_or_last or split_point in self.lecture.video_segments:
            # do not split here ...
            return False
        else:
            # add the split point at the current location
            self.lecture.video_segments.append(split_point)
            # keep sorted
            self.lecture.video_segments = sorted(self.lecture.video_segments)

            if add_undo:
                self.undo_stack.append({
                    "operation": "vid_seg_split",
                    "split_point": split_point,
                })

            return True

    def segment_merge(self, split_point, add_undo):
        if split_point in self.lecture.video_segments:
            self.lecture.video_segments.remove(split_point)

            if add_undo:
                self.undo_stack.append({
                    "operation": "vid_seg_merge",
                    "split_point": split_point,
                })

            return True
        else:
            return False

    def btn_vid_seg_split_click(self, button):
        if self.segment_split(self.last_video_frame, True):
            self.update_video_segment_buttons()

    def btn_vid_seg_merge_click(self, button):
        if self.segment_merge(self.last_video_frame, True):
            self.update_video_segment_buttons()

    def segment_keyframe_add(self, frame_index, add_undo):
        if frame_index not in self.lecture.video_segment_keyframes:
            # add the key-frame at the current location
            self.lecture.video_segment_keyframes.append(frame_index)
            # keep sorted
            self.lecture.video_segment_keyframes = sorted(self.lecture.video_segment_keyframes)

            if add_undo:
                self.undo_stack.append({
                    "operation": "vid_seg_keyframe_add",
                    "frame_index": frame_index,
                })

            return True
        else:
            return False

    def segment_keyframe_del(self, frame_index, add_undo):
        if frame_index in self.lecture.video_segment_keyframes:
            self.lecture.video_segment_keyframes.remove(frame_index)

            if add_undo:
                self.undo_stack.append({
                    "operation": "vid_seg_keyframe_del",
                    "frame_index": frame_index,
                })

            return True
        else:
            return False

    def btn_vid_seg_keyframe_add_click(self, button):
        if self.segment_keyframe_add(self.last_video_frame, True):
            self.update_video_segment_buttons()

    def btn_vid_seg_keyframe_del_click(self, button):
        if self.segment_keyframe_del(self.last_video_frame, True):
            self.update_video_segment_buttons()

    def update_kf_tag_buttons(self, current_object):
        #  determine all labels used so far ...
        all_labels = current_object.all_unique_labels()
        sorted_labels = [(all_labels[tag], tag) for tag in all_labels if tag is not None]
        sorted_labels.sort(key=lambda x:x[0], reverse=True)

        for btn_idx in range(1, len(self.buttons_prev_segment_labels)):
            current_button = self.buttons_prev_segment_labels[btn_idx]
            if btn_idx - 1 >= len(sorted_labels):
                # not that many tags used so far ...
                current_button.visible = False
            else:
                # update with next most common tag
                current_button.tag = sorted_labels[btn_idx - 1][1]
                current_button.updateText(sorted_labels[btn_idx - 1][1])
                current_button.visible = True

    def btn_keyframe_label_set_click(self, button):
        # first, check an object is selected on the list
        selected_name = self.object_selector.selected_option_value
        if selected_name is None:
            # nothing is selected
            print("No object is selected")
            return

        # current_frame = self.last_video_frame
        # current_time = self.last_video_time
        # selected_name = self.canvas.selected_element

        current_object = self.lecture[selected_name]
        current_object_loc = current_object.get_location_at(self.last_video_frame, False, False)

        if current_object_loc is None:
            # no key-frame to label ...
            # add a key-frame first ...
            self.btn_keyframe_add_click(self.btn_keyframe_add)

            # recover location again ...
            current_object_loc = current_object.get_location_at(self.last_video_frame, False, False)

            if current_object_loc is None:
                print("Object Key-frame was not added, A label cannot be set")
                return

        current_text = "" if current_object_loc.label is None else current_object_loc.label

        self.update_kf_tag_buttons(current_object)

        # set rename mode
        self.prepare_confirm_input_mode(5, current_text, None)

    def btn_prev_segment_click(self, button):
        if button.tag is None:
            self.txt_object_name.updateText("")
        else:
            self.txt_object_name.updateText(button.tag)

        self.btn_text_operation_accept_click(button)

    def btn_export_segments_click(self, button):
        # force pause the video???
        self.btn_pause_click(None)

        # check if output directory exists
        main_path = self.output_prefix
        if not os.path.exists(main_path):
            os.mkdir(main_path)

        # check if keyframes sub-directory exists
        keyframes_path = main_path + "/keyframes"
        if not os.path.exists(keyframes_path):
            os.mkdir(keyframes_path)

        # save images for key-frames ....
        current_frame = self.last_video_frame
        frame_times = []
        for keyframe_idx in self.lecture.video_segment_keyframes:
            # use the player to extract the video frame ...
            self.player.set_player_frame(keyframe_idx, False)
            frame_img, frame_idx = self.player.video_player.get_frame()

            # keep the key-frame absolute times
            frame_times.append(self.player.video_player.play_abs_position)

            # save image to file ...
            cv2.imwrite(keyframes_path + "/" + str(frame_idx) + ".png", frame_img)

        # restore player current location ...
        self.player.set_player_frame(current_frame, False)

        # save XML string to output file
        xml_data = self.lecture.generate_export_xml(frame_times)

        export_filename = main_path + "/segments.xml"
        out_file = open(export_filename, "w")
        out_file.write(xml_data)
        out_file.close()

        print("Metadata Saved to: " + export_filename)

    def btn_export_object_click(self, button):
        selected_name = self.canvas.selected_element

        if selected_name is None:
            print("No Object selected")
            return

        current_object = self.lecture[selected_name]
        total_frames = int(self.player.video_player.total_frames)

        output_name = "{0:s}_{1:s}.csv".format(self.output_prefix, selected_name)
        current_object.export_CSV(output_name, total_frames)

        print("Object data saved to " + output_name)

    def player_click(self, player_object, button, pos):
        # adjust coordinate from player space to canvas coordinate ..
        adjusted_px = pos[0] + (self.player.position[0] - self.canvas.position[0])
        adjusted_py = pos[1] + (self.player.position[1] - self.canvas.position[1])

        if self.text_operation == -1 and button == 1:
            self.user_selected_position = (adjusted_px, adjusted_py)
            self.prepare_confirm_input_mode(1, "", None)
        if not self.canvas.locked and button == 3:
            # find closest object (in time) which contains current point
            if self.player.video_player.zoom_factor == 0:
                adjusted_point = (adjusted_px, adjusted_py)
            else:
                raw_point = np.array([adjusted_px, adjusted_py])

                translation, scale = self.compute_canvas_zoom_translation_scale()
                scaled_point = (raw_point - translation) / scale
                adjusted_point = scaled_point.tolist()

            current_frame_idx = self.player.video_player.last_frame_idx
            closest_object = self.lecture.find_temporal_closest_point_container(adjusted_point, current_frame_idx)
            if closest_object is not None:
                self.select_object(closest_object.id, 3)

    def main_key_up(self, scancode, key, unicode):
        if self.container_text_input.visible:
            # only use these keyboard shortcuts in the "Normal" mode ...
            return

        # Key assignments .....
        if key == 32:
            # SPACE BAR
            if self.button_play.visible:
                self.btn_play_click(self.button_play)
            else:
                self.btn_pause_click(self.button_pause)
        elif key == 122:
            # Z
            # -1
            self.btn_change_frame(self.precision_buttons[-1])
        elif key == 120:
            # X
            # +1
            self.btn_change_frame(self.precision_buttons[1])
        elif key == 97:
            # A
            # -10
            self.btn_change_frame(self.precision_buttons[-10])
        elif key == 115:
            # S
            # +10
            self.btn_change_frame(self.precision_buttons[10])
        elif key == 113:
            # Q
            # -100
            self.btn_change_frame(self.precision_buttons[-100])
        elif key == 119:
            # W
            # +100
            self.btn_change_frame(self.precision_buttons[100])
        elif key == 49:
            # 1
            # -1000
            self.btn_change_frame(self.precision_buttons[-1000])
        elif key == 50:
            # 2
            # +1000
            self.btn_change_frame(self.precision_buttons[1000])
        elif key == 99:
            # C
            # (speed * 0.5)
            self.btn_dec_speed_click(self.btn_dec_speed)
        elif key == 118:
            # V
            # (speed * 2.0)
            self.btn_inc_speed_click(self.btn_inc_speed)
        elif key == 101:
            # e
            # SET -> object status
            if self.btn_keyframe_label_set.visible:
                self.btn_keyframe_label_set_click(self.btn_keyframe_label_set)
        elif key == 107:
            # k
            # if there is object selected... add keyframe
            if self.btn_keyframe_add.visible:
                self.btn_keyframe_add_click(self.btn_keyframe_add)
        elif key == 44:
            # <
            if self.container_keyframe_options.visible:
                # copy previous key-frame if possible
                self.copy_neighbor_keyframe_location(False)
        elif key == 46:
            # >
            if self.container_keyframe_options.visible:
                # copy next key-frame if possible
                self.copy_neighbor_keyframe_location(True)

        # print((scancode, key))

    def panning_hor_scroll_change(self, scroll):
        self.player.video_player.set_horizontal_panning(scroll.value / 100.0)
        self.update_canvas_objects()

    def panning_ver_scroll_change(self, scroll):
        self.player.video_player.set_vertical_panning(scroll.value / 100.0)
        self.update_canvas_objects()

    def btn_inc_zoom_click(self, button):
        if self.player.video_player.zoom_increase():
            self.update_zoom_options()

    def btn_dec_zoom_click(self, button):
        if self.player.video_player.zoom_decrease():
            self.update_zoom_options()

    def update_zoom_options(self):
        if self.player.video_player.zoom_factor == 0:
            self.label_zoom_current.set_text("Zoom: 100%")
            self.panning_hor_scroll.visible = False
            self.panning_ver_scroll.visible = False
        else:
            self.label_zoom_current.set_text("Zoom: " + str(pow(2, self.player.video_player.zoom_factor)) + "00%")
            self.panning_hor_scroll.visible = True
            self.panning_ver_scroll.visible = True

        # update canvas!!!
        self.update_canvas_objects()

    def btn_object_split_poly_click(self, button):
        selected_name = self.canvas.selected_element
        current_object = self.lecture[selected_name]

        tempo_object = current_object.make_polygon_split_copy()

        # remove selected object .... using simulated GUI events ...
        self.text_operation = 3
        self.btn_text_operation_accept_click(self.btn_text_operation_accept)

        # add new object ...
        first_loc = tempo_object.locations[0]
        self.add_object(tempo_object.id, tempo_object.name, first_loc.frame, first_loc.abs_time,
                        tempo_object.shape_type, first_loc.polygon_points)

        new_object = self.lecture[selected_name]
        new_object.locations[0].label = first_loc.label
        new_object.locations[0].visible = first_loc.visible

        tempo_location = VideoObjectLocation.fromLocation(new_object.locations[0])
        self.undo_stack.append({
            "operation": "object_added",
            "id": tempo_object.id,
            "name": tempo_object.name,
            "shape": tempo_object.shape_type,
            "location": tempo_location,
        })

        # copy all existing key-frames after the first one
        for tempo_loc in tempo_object.locations[1:]:
            new_object.set_location_at(tempo_loc.frame, tempo_loc.abs_time, tempo_loc.visible, tempo_loc.polygon_points)

            self.undo_stack.append({
                "operation": "keyframe_added",
                "object_id": selected_name,
                "old_location": None,
                "new_location": VideoObjectLocation.fromLocation(tempo_loc),
            })

        self.update_canvas_objects()
        self.update_keyframe_buttons()

