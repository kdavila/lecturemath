

from AM_CommonTools.util.time_helper import TimeHelper
from AM_CommonTools.interface.controls.screen import Screen
from AM_CommonTools.interface.controls.screen_button import ScreenButton
from AM_CommonTools.interface.controls.screen_canvas import ScreenCanvas
from AM_CommonTools.interface.controls.screen_container import ScreenContainer
from AM_CommonTools.interface.controls.screen_horizontal_scroll import ScreenHorizontalScroll
from AM_CommonTools.interface.controls.screen_label import ScreenLabel
from AM_CommonTools.interface.controls.screen_textbox import ScreenTextbox
from AM_CommonTools.interface.controls.screen_textlist import ScreenTextlist
from AM_CommonTools.interface.controls.screen_video_player import ScreenVideoPlayer

from AccessMath.util.misc_helper import MiscHelper

class ST3D_Visualizer(Screen):
    def __init__(self, size, estimator_filename, structure_filename, db_name, lecture_title, forced_resolution=None):
        Screen.__init__(self, "Spatio-Temporal 3D Structure Visualization", size)

        general_background = (80, 80, 95)
        darker_background = (70, 70, 85)

        self.db_name = db_name
        self.lecture_title = lecture_title

        frame_times, frame_indices, estimator = MiscHelper.dump_load(estimator_filename)
        self.cc_stability = estimator
        self.ST3D_struct = MiscHelper.dump_load(structure_filename)

        player_type = ScreenVideoPlayer.VideoPlayerST3D

        # main video player
        self.player = ScreenVideoPlayer("video_player", 960, 540)
        self.player.position = (50, 50)
        self.player.open_video_files((self.cc_stability, self.ST3D_struct), forced_resolution, player_type)
        self.player.frame_changed_callback = self.video_frame_change
        self.player.mouse_leave_callback = self.player_mouse_left
        self.player.mouse_motion_callback = self.player_mouse_moved
        self.player.click_callback = self.player_mouse_clicked
        self.player.play()
        self.elements.append(self.player)

        print("Total Video Length: " + TimeHelper.secondsToStr(self.player.video_player.total_length / 1000))
        print("Total Video Frames: " + str(self.player.video_player.total_frames))

        # canvas used for annotations
        self.canvas = ScreenCanvas("canvas", 1040, 620)
        self.canvas.position = (10, 10)
        self.canvas.locked = True
        self.canvas.object_edited_callback = self.canvas_object_edited
        self.canvas.object_selected_callback = self.canvas_selection_changed
        self.elements.append(self.canvas)

        self.canvas.add_rectangle_element("loc_cc", 0, 0, 50, 50, custom_color=(0, 255, 0))
        self.canvas.add_rectangle_element("loc_group", 0, 50, 50, 50, custom_color=(255, 255, 0))
        self.canvas.elements["loc_cc"].visible = False
        self.canvas.elements["loc_group"].visible = False

        self.last_video_frame = None
        self.last_video_time = None
        self.last_mouse_position = None

        # EXIT BUTTON
        exit_button = ScreenButton("exit_button", "EXIT", 16, 70, 0)
        exit_button.set_colors((192, 255, 128), (64, 64, 64))
        exit_button.position = (self.width - exit_button.width - 15, self.height - exit_button.height - 15)
        exit_button.click_callback = self.close_click
        self.elements.append(exit_button)

        # video controllers
        self.container_video_controls = ScreenContainer("container_video_controls", (1050, 130), darker_background)
        self.container_video_controls.position = (5, self.canvas.get_bottom() + 5)
        self.elements.append(self.container_video_controls)

        step_1 = self.player.video_player.total_frames / 100
        self.position_scroll = ScreenHorizontalScroll("video_position", 0, self.player.video_player.total_frames - 1, 0,
                                                      step_1)
        self.position_scroll.position = (5, 5)
        self.position_scroll.width = 1040
        self.position_scroll.scroll_callback = self.main_scroll_change
        self.container_video_controls.append(self.position_scroll)

        # Frame count
        self.label_frame_count = ScreenLabel("frame_count", "Frame Count: " + str(int(self.player.video_player.total_frames)), 18)
        self.label_frame_count.position = (15, self.position_scroll.get_bottom() + 10)
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
        dec_speed = ScreenButton("dec_speed", "0.5x", 16, 70, 0)
        dec_speed.set_colors((192, 255, 128), (64, 64, 64))
        dec_speed.position = (self.label_player_speed.get_left() - dec_speed.width - 15, self.label_player_speed.get_top())
        dec_speed.click_callback = self.btn_dec_speed_click
        self.container_video_controls.append(dec_speed)

        inc_speed = ScreenButton("inc_speed", "2.0x", 16, 70, 0)
        inc_speed.set_colors((192, 255, 128), (64, 64, 64))
        inc_speed.position = (self.label_player_speed.get_right() + 15, self.label_player_speed.get_top())
        inc_speed.click_callback = self.btn_inc_speed_click
        self.container_video_controls.append(inc_speed)

        # Precision buttons ....
        v_pos = self.label_time_current.get_bottom() + 15
        btn_w = 70
        for idx, value in enumerate([-1000, -100, -10, -1]):
            prec_button = ScreenButton("prec_button_m_" + str(idx), str(value), 16, btn_w, 0)
            prec_button.set_colors((192, 255, 128), (64, 64, 64))
            prec_button.position = (15 + idx * (btn_w + 15) , v_pos)
            prec_button.click_callback = self.btn_change_frame
            prec_button.tag = value
            self.container_video_controls.append(prec_button)

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
            prec_button.position = (15 + (5 + idx) * (btn_w + 15) , v_pos)
            prec_button.click_callback = self.btn_change_frame
            prec_button.tag = value
            self.container_video_controls.append(prec_button)

        self.elements.back_color = general_background

        # Zoom controls ....
        options_btn_width = 120
        options_padding = 15
        options_left = self.canvas.get_right() + options_padding
        options_width = self.width - self.canvas.get_right() - options_padding * 2

        self.container_zoom = ScreenContainer("container_zoom", (options_width, 170), darker_background)
        self.container_zoom.position = (options_left, options_padding)
        self.elements.append(self.container_zoom)

        # increase zoom
        self.btn_video_zoom_inc = ScreenButton("btn_video_zoom_inc", "Zoom In", 26, options_btn_width)
        self.btn_video_zoom_inc.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_video_zoom_inc.position = (int(options_width * 0.25 - options_btn_width * 0.5), 10)
        self.btn_video_zoom_inc.click_callback = self.btn_video_zoom_inc_click
        self.container_zoom.append(self.btn_video_zoom_inc)

        # decrease zoom
        self.btn_video_zoom_dec = ScreenButton("btn_video_zoom_dec", "Zoom Out", 26, options_btn_width)
        self.btn_video_zoom_dec.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_video_zoom_dec.position = (int(options_width * 0.75 - options_btn_width * 0.5), 10)
        self.btn_video_zoom_dec.click_callback = self.btn_video_zoom_dec_click
        self.container_zoom.append(self.btn_video_zoom_dec)

        # horizontal panning
        # ... label ...
        self.label_padding_x = ScreenLabel("label_padding_x", "X", 22, max_width=options_width - 20)
        self.label_padding_x.background = darker_background
        self.label_padding_x.position = (10, self.btn_video_zoom_inc.get_bottom() + 10)
        self.label_padding_x.set_color((255, 255, 255))
        self.container_zoom.append(self.label_padding_x)

        # ... scroll ....
        self.scroll_padding_x = ScreenHorizontalScroll("scroll_padding_x", 0, 100, 0, 10)
        self.scroll_padding_x.position = (10, self.label_padding_x.get_bottom() + 10)
        self.scroll_padding_x.width = options_width - 20
        self.scroll_padding_x.scroll_callback = self.scroll_padding_x_change
        self.container_zoom.append(self.scroll_padding_x)

        # vertical panning
        # ... label ...
        self.label_padding_y = ScreenLabel("label_padding_y", "Y", 22, max_width=options_width - 20)
        self.label_padding_y.background = darker_background
        self.label_padding_y.position = (10, self.scroll_padding_x.get_bottom() + 10)
        self.label_padding_y.set_color((255, 255, 255))
        self.container_zoom.append(self.label_padding_y)

        # ... scroll ....
        self.scroll_padding_y = ScreenHorizontalScroll("scroll_padding_y", 0, 100, 0, 10)
        self.scroll_padding_y.position = (10, self.label_padding_y.get_bottom() + 10)
        self.scroll_padding_y.width = options_width - 20
        self.scroll_padding_y.scroll_callback = self.scroll_padding_y_change
        self.container_zoom.append(self.scroll_padding_y)

        # ================================
        # visualization modes .....
        self.container_vis_mode = ScreenContainer("container_vis_mode", (options_width, 80), darker_background)
        self.container_vis_mode.position = (options_left, self.container_zoom.get_bottom() + options_padding)
        self.elements.append(self.container_vis_mode)

        self.btn_vis_mode_binary = ScreenButton("btn_vis_mode_binary", "Binary", 26, options_btn_width)
        self.btn_vis_mode_binary.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_vis_mode_binary.position = (int(options_width * 0.1667 - options_btn_width * 0.5), 10)
        self.btn_vis_mode_binary.click_callback = self.btn_vis_mode_binary_click
        self.container_vis_mode.append(self.btn_vis_mode_binary)

        self.btn_vis_mode_stable = ScreenButton("btn_vis_mode_stable", "Stable CC", 26, options_btn_width)
        self.btn_vis_mode_stable.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_vis_mode_stable.position = (int(options_width * 0.5 - options_btn_width * 0.5), 10)
        self.btn_vis_mode_stable.click_callback = self.btn_vis_mode_stable_click
        self.container_vis_mode.append(self.btn_vis_mode_stable)

        self.btn_vis_mode_reconstructed = ScreenButton("btn_vis_mode_reconstructed", "Clean", 26, options_btn_width)
        self.btn_vis_mode_reconstructed.set_colors((192, 255, 128), (64, 64, 64))
        self.btn_vis_mode_reconstructed.position = (int(options_width * 0.8333 - options_btn_width * 0.5), 10)
        self.btn_vis_mode_reconstructed.click_callback = self.btn_vis_mode_reconstructed_click
        self.container_vis_mode.append(self.btn_vis_mode_reconstructed)

        # ================================
        # visualization modes .....
        self.container_info = ScreenContainer("container_info", (options_width, 300), darker_background)
        self.container_info.position = (options_left, self.container_vis_mode.get_bottom() + options_padding)
        self.elements.append(self.container_info)

        self.lbl_current_info = ScreenLabel("lbl_current_info", "[]", 26, max_width=options_width - options_padding * 2,
                                            centered=0)
        self.lbl_current_info.position = (options_padding, options_padding)
        self.lbl_current_info.set_color((255, 255, 255))
        self.lbl_current_info.set_background(darker_background)
        self.container_info.append(self.lbl_current_info)

    def video_frame_change(self, next_frame, next_abs_time):
        # update the scroll bar
        self.position_scroll.value = next_frame

        self.last_video_frame = next_frame
        self.last_video_time = next_abs_time

        self.label_frame_current.set_text("Current frame: " + str(next_frame))
        self.label_time_current.set_text("Current time: " + TimeHelper.stampToStr(next_abs_time))

        # update canvas ...
        self.update_canvas_objects()

        # update Info ....
        if self.last_mouse_position is not None:
            self.player_mouse_moved(self.player, self.last_mouse_position, (0, 0), (0, 0, 0))


    def canvas_object_edited(self, canvas, object_name):
        # do nothing for now ....
        pass

    def canvas_selection_changed(self, object_selected):
        # do nothing for now ....
        pass

    def close_click(self, button):
        self.return_screen = None
        print("APPLICATION FINISHED")

    def main_scroll_change(self, scroll):
        self.player.set_player_frame(int(scroll.value), True)

    def update_canvas_objects(self):
        pass

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

    def btn_video_zoom_inc_click(self, button):
        self.player.video_player.zoom_increase()

    def btn_video_zoom_dec_click(self, button):
        self.player.video_player.zoom_decrease()

    def scroll_padding_x_change(self, scroll):
        self.player.video_player.set_horizontal_panning(scroll.value / 100.0)

    def scroll_padding_y_change(self, scroll):
        self.player.video_player.set_vertical_panning(scroll.value / 100.0)

    def btn_vis_mode_binary_click(self, button):
        self.player.video_player.set_binary_mode()

    def btn_vis_mode_reconstructed_click(self, button):
        self.player.video_player.set_reconstructed_mode()

    def btn_vis_mode_stable_click(self, button):
        self.player.video_player.set_stable_cc_mode()

    def player_mouse_left(self, player, pos, rel, buttons):
        self.lbl_current_info.set_text("")
        self.last_mouse_position = None
        self.canvas.elements["loc_cc"].visible = False
        self.canvas.elements["loc_group"].visible = False

    def get_mouse_cc_info(self, pixel_x, pixel_y):
        try:
            frame_offset = self.ST3D_struct.frame_indices.index(self.last_video_frame)
        except:
            return None, None

        # get the original local CC
        current_frame_ccs = self.cc_stability.cc_idx_per_frame[frame_offset]

        pixel_cc_idx = -1
        cc_instance_info = -1
        pixel_cc_instance = None
        for cc_global_idx, cc in current_frame_ccs:
            if (cc.min_x <= pixel_x <= cc.max_x and
                    cc.min_y <= pixel_y <= cc.max_y and
                    cc.img[pixel_y - cc.min_y, pixel_x - cc.min_x] > 0):
                # Global CC found!
                pixel_cc_idx = cc_global_idx
                pixel_cc_instance = cc

                cc_instance_count = len(self.cc_stability.unique_cc_frames[cc_global_idx])
                cc_instance_first_offset = self.cc_stability.unique_cc_frames[cc_global_idx][0][0]
                cc_instance_first = self.ST3D_struct.frame_indices[cc_instance_first_offset]
                cc_instance_last_offset = self.cc_stability.unique_cc_frames[cc_global_idx][-1][0]
                cc_instance_last = self.ST3D_struct.frame_indices[cc_instance_last_offset]
                cc_instance_info = (cc_instance_count, cc_instance_first, cc_instance_last)
                break

        if pixel_cc_idx >= 0:
            local_cc_info = (pixel_cc_idx, pixel_cc_instance, cc_instance_info)
        else:
            local_cc_info = None

        # Get the grouped CC ....
        groups_in_frame = self.ST3D_struct.groups_in_frame_range(self.last_video_frame, self.last_video_frame)
        groups_in_region = self.ST3D_struct.groups_in_space_region(pixel_x, pixel_x, pixel_y, pixel_y, groups_in_frame)

        # check for each group if it has this pixel in this frame ...
        current_group_idx = -1
        for group_idx in groups_in_region:
            group_ages = self.ST3D_struct.cc_group_ages[group_idx]
            group_images = self.ST3D_struct.cc_group_images[group_idx]

            g_min_x, g_max_x, g_min_y, g_max_y = self.ST3D_struct.cc_group_boundaries[group_idx]
            rel_pixel_x = pixel_x - g_min_x
            rel_pixel_y = pixel_y - g_min_y

            for age_idx in range(len(group_ages) - 1):
                if (self.ST3D_struct.frame_indices[group_ages[age_idx]] <= self.last_video_frame
                        <= self.ST3D_struct.frame_indices[group_ages[age_idx + 1]]):
                    # found the interval that covers the frame ... it either has the pixel or not ...
                    if group_images[age_idx][rel_pixel_y, rel_pixel_x] > 0:
                        # found!
                        current_group_idx = group_idx

                    # no further check for this candidate is required ...
                    break

            if current_group_idx != -1:
                # already found!
                break

        if current_group_idx != -1:
            group_ages = [self.ST3D_struct.frame_indices[age_offset]
                          for age_offset in self.ST3D_struct.cc_group_ages[current_group_idx]]
            group_boundaries = self.ST3D_struct.cc_group_boundaries[current_group_idx]
            grouped_cc_info = (current_group_idx, group_ages, group_boundaries)
        else:
            grouped_cc_info = None

        return local_cc_info, grouped_cc_info

    def player_mouse_moved(self, player, pos, rel, buttons):
        self.last_mouse_position = pos

        # convert position to absolute (x, y) coord in video space ...
        rel_render_x = pos[0] / self.player.render_width
        rel_render_y = pos[1] / self.player.render_height

        if 0.0 <= rel_render_x <= 1.0 and 0.0 <= rel_render_y <= 1.0:
            offset_x = self.player.video_player.visible_left()
            offset_y = self.player.video_player.visible_top()
            visible_w = self.player.video_player.visible_width()
            visible_h = self.player.video_player.visible_height()

            canvas_delta_x = self.player.render_location[0] - self.canvas.position[0]
            canvas_delta_y = self.player.render_location[1] - self.canvas.position[1]

            pixel_x = int(offset_x + rel_render_x * visible_w)
            pixel_y = int(offset_y + rel_render_y * visible_h)

            # find the current frame offset ...
            local_cc_info, grouped_cc_info = self.get_mouse_cc_info(pixel_x, pixel_y)

            output_text = "(X, Y) = ({0:d}, {1:d})\n\n".format(pixel_x, pixel_y)
            loc_cc = self.canvas.elements["loc_cc"]
            if local_cc_info is not None:
                pixel_cc_idx, cc_instance, cc_instance_info = local_cc_info
                cc_instance_count, cc_instance_first, cc_instance_last = cc_instance_info

                output_text += "Global CC ID: {0:d}\n".format(pixel_cc_idx)
                output_text += str(cc_instance) + "\n"
                output_text += " T: [{0:d}, {1:d}]\n".format(cc_instance_first, cc_instance_last)
                output_text += "-> Instance Count: {0:d}\n\n".format(cc_instance_count)

                loc_cc.x = ((cc_instance.min_x - offset_x) / visible_w) * self.player.render_width + canvas_delta_x
                loc_cc.y = ((cc_instance.min_y - offset_y) / visible_h) * self.player.render_height + canvas_delta_y
                loc_cc.w = ((cc_instance.max_x - cc_instance.min_x + 1) / visible_w) * self.player.render_width
                loc_cc.h = ((cc_instance.max_y - cc_instance.min_y + 1) / visible_h) * self.player.render_height

                loc_cc.visible = True
            else:
                loc_cc.visible = False

            loc_group = self.canvas.elements["loc_group"]
            if grouped_cc_info is not None:
                current_group_idx, group_ages, group_boundaries = grouped_cc_info
                g_min_x, g_max_x, g_min_y, g_max_y = group_boundaries

                output_text += "CC GROUP ID: {0:d}\n".format(current_group_idx)
                output_text += " -> X: [{0:d}, {1:d}]\n".format(g_min_x, g_max_x)
                output_text += " -> Y: [{0:d}, {1:d}]\n".format(g_min_y, g_max_y)
                output_text += " -> T: [{0:d}, {1:d}]\n\n".format(group_ages[0], group_ages[-1])

                loc_group.x = ((g_min_x - offset_x) / visible_w) * self.player.render_width + canvas_delta_x
                loc_group.y = ((g_min_y - offset_y) / visible_h) * self.player.render_height + canvas_delta_y
                loc_group.w = ((g_max_x - g_min_x + 1) / visible_w) * self.player.render_width
                loc_group.h = ((g_max_y - g_min_y + 1) / visible_h) * self.player.render_height

                loc_group.visible = True
            else:
                loc_group.visible = False

            self.lbl_current_info.set_text(output_text)

    def player_mouse_clicked(self, player):
        print(self.lbl_current_info.text)
