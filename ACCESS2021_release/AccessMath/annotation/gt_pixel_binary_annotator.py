
import cv2
import math
import time
import numpy as np

from AM_CommonTools.interface.controls.screen import Screen
from AM_CommonTools.interface.controls.screen_label import ScreenLabel
from AM_CommonTools.interface.controls.screen_button import ScreenButton
from AM_CommonTools.interface.controls.screen_image import ScreenImage
from AM_CommonTools.interface.controls.screen_container import ScreenContainer
from AM_CommonTools.interface.controls.screen_horizontal_scroll import ScreenHorizontalScroll

from AccessMath.preprocessing.content.labeler import Labeler


class GTPixelBinaryAnnotator(Screen):
    ModeNavigation = 0
    ModeEdition = 1
    ModeConfirmCancel = 2
    ModeGrowCC_Select = 3
    ModeGrowCC_Growing = 4
    ModeShrinkCC_Select = 5
    ModeShrinkCC_Shrinking = 6
    ModeDeleteCC_Select = 7
    ModePixelwise_Displacing = 8

    ViewModeRaw = 0
    ViewModeGray = 1
    ViewModeBinary = 2
    ViewModeSoftCombined = 3
    ViewModeHardCombined = 4

    CCShowBlack = 0
    CCShowColored = 1
    CCShowColors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                    (255, 255, 0), (255, 0, 255), (0, 255, 255),
                    (128, 0, 0), (0, 128, 0), (0, 0, 128),
                    (128, 128, 0), (128, 0, 128), (0, 128, 128),
                    (255, 128, 0), (255, 0, 128), (0, 255, 128),
                    (128, 255, 0), (128, 0, 255), (0, 128, 255),]

    CCExpansionDistance = 1

    PlusMinusMultiTime = 0.40
    PlusMinusMultiFactor = 5

    def __init__(self, size, raw_input, binary_input, dark_mode, parent_screen=None):
        Screen.__init__(self, "Ground Truth Binary Pixel Annotation Interface", size)

        self.small_mode = self.height < 800

        # base images
        self.base_raw = raw_input.copy()
        tempo_gray = cv2.cvtColor(raw_input, cv2.COLOR_RGB2GRAY)
        if dark_mode:
            tempo_gray = 255 - tempo_gray

        self.base_gray = np.zeros((self.base_raw.shape[0], self.base_raw.shape[1], 3), dtype=np.uint8)
        self.base_gray[:, :, 0] = tempo_gray.copy()
        self.base_gray[:, :, 1] = tempo_gray.copy()
        self.base_gray[:, :, 2] = tempo_gray.copy()

        self.base_binary = binary_input.copy()

        self.base_ccs = None
        self.base_cc_image = None    # colored version

        # stats
        self.stats_pixels_white = 0
        self.stats_pixels_black = 0
        self.stats_pixels_ratio = 0

        self.stats_cc_count = 0
        self.stats_cc_size_min = 0
        self.stats_cc_size_max = 0
        self.stats_cc_size_mean = 0
        self.stats_cc_size_median = 0

        # for automatic CC expansion
        self.selected_CC = None
        self.CC_expansion_pixels_light = None
        self.CC_expansion_pixels_dark = None
        self.count_expand_light = 0
        self.count_expand_dark = 0
        # ... shrinking ...
        self.CC_shrinking_pixels_light = None
        self.CC_shrinking_pixels_dark = None
        self.count_shrink_light = 0
        self.count_shrink_dark = 0

        # automatic CC displacement
        self.displacement_is_vertical = None
        self.displacement_pos_remove = None
        self.displacement_pos_add = None
        self.displacement_neg_remove = None
        self.displacement_neg_add = None
        self.displacement_pixels_removing = None
        self.displacement_pixels_adding = None
        self.displacement_percentage = None

        # view params
        self.view_scale = 1.0
        self.max_scale = None
        self.view_mode = GTPixelBinaryAnnotator.ViewModeBinary
        self.editor_mode = GTPixelBinaryAnnotator.ModeNavigation
        self.cc_show_mode = GTPixelBinaryAnnotator.CCShowColored
        self.min_highlight_size = 0

        # appearance parameters
        general_background = (125, 40, 20)
        text_color = (255, 255, 255)
        button_text_color = (50, 35, 20)
        button_back_color = (228, 228, 228)
        self.elements.back_color = general_background

        # add elements....
        # right panel button size and horizontal locations
        container_width = 330

        button_width = 190
        button_left = (container_width - button_width) / 2

        button_2_width = 150
        button_2_left = int(container_width * 0.25) - button_2_width / 2
        button_2_right = int(container_width * 0.75) - button_2_width / 2

        button_3_width = 100
        button_3_left = 10
        button_3_middle = (container_width - button_3_width) / 2
        button_3_right = container_width - button_3_width - 10

        button_4_width = 75
        button_4_left_1 = int(container_width * 0.125) - button_4_width / 2
        button_4_left_2 = int(container_width * 0.375) - button_4_width / 2
        button_4_left_3 = int(container_width * 0.625) - button_4_width / 2
        button_4_left_4 = int(container_width * 0.875) - button_4_width / 2

        # View panel with Zoom control buttons
        self.container_zoom_buttons = ScreenContainer("container_zoom_buttons", (container_width, 80),
                                                      back_color=general_background)
        self.container_zoom_buttons.position = (self.width - container_width - 10, 10)
        self.elements.append(self.container_zoom_buttons)

        # zoom ....
        self.lbl_zoom = ScreenLabel("lbl_zoom", "Zoom: 100%", 21, container_width - 10, 1)
        self.lbl_zoom.position = (5, 5)
        self.lbl_zoom.set_background(general_background)
        self.lbl_zoom.set_color(text_color)
        self.container_zoom_buttons.append(self.lbl_zoom)

        self.btn_zoom_reduce = ScreenButton("btn_zoom_reduce", "[ - ]", 21, 90)
        self.btn_zoom_reduce.set_colors(button_text_color, button_back_color)
        self.btn_zoom_reduce.position = (10, self.lbl_zoom.get_bottom() + 10)
        self.btn_zoom_reduce.click_callback = self.btn_zoom_reduce_click
        self.container_zoom_buttons.append(self.btn_zoom_reduce)

        self.btn_zoom_increase = ScreenButton("btn_zoom_increase", "[ + ]", 21, 90)
        self.btn_zoom_increase.set_colors(button_text_color, button_back_color)
        self.btn_zoom_increase.position = (self.container_zoom_buttons.width - self.btn_zoom_increase.width - 10,
                                           self.lbl_zoom.get_bottom() + 10)
        self.btn_zoom_increase.click_callback = self.btn_zoom_increase_click
        self.container_zoom_buttons.append(self.btn_zoom_increase)

        self.btn_zoom_zero = ScreenButton("btn_zoom_zero", "100%", 21, 90)
        self.btn_zoom_zero.set_colors(button_text_color, button_back_color)
        self.btn_zoom_zero.position = ((self.container_zoom_buttons.width - self.btn_zoom_zero.width) / 2,
                                       self.lbl_zoom.get_bottom() + 10)
        self.btn_zoom_zero.click_callback = self.btn_zoom_zero_click
        self.container_zoom_buttons.append(self.btn_zoom_zero)

        # ===========================
        self.container_cc_expansion = ScreenContainer("container_cc_expansion", (container_width, 220), back_color=general_background)
        self.container_cc_expansion.position = (self.container_zoom_buttons.get_left(), self.container_zoom_buttons.get_bottom() + 10)
        self.elements.append(self.container_cc_expansion)

        self.lbl_cc_expansion = ScreenLabel("lbl_cc_expansion", "Expanding CC", 21, container_width - 10, 1)
        self.lbl_cc_expansion.position = (5, 5)
        self.lbl_cc_expansion.set_background(general_background)
        self.lbl_cc_expansion.set_color(text_color)
        self.container_cc_expansion.append(self.lbl_cc_expansion)

        self.lbl_cc_expansion_lighter = ScreenLabel("lbl_cc_expansion_lighter",
                                                    "Lighter Pixels = " + str(self.count_expand_light), 21,
                                                    container_width - 10, 1)
        self.lbl_cc_expansion_lighter.position = (5, self.lbl_cc_expansion.get_bottom() + 10)
        self.lbl_cc_expansion_lighter.set_background(general_background)
        self.lbl_cc_expansion_lighter.set_color(text_color)
        self.container_cc_expansion.append(self.lbl_cc_expansion_lighter)

        self.expansion_scroll_lighter = ScreenHorizontalScroll("expansion_scroll_lighter", 0, 100, 0, 10)
        self.expansion_scroll_lighter.position = (5, self.lbl_cc_expansion_lighter.get_bottom() + 10)
        self.expansion_scroll_lighter.width = container_width - 10
        self.expansion_scroll_lighter.scroll_callback = self.expansion_scroll_lighter_change
        self.container_cc_expansion.append(self.expansion_scroll_lighter)

        self.lbl_cc_expansion_darker = ScreenLabel("lbl_cc_expansion_darker",
                                                    "Darker Pixels = " + str(self.count_expand_dark), 21,
                                                    container_width - 10, 1)
        self.lbl_cc_expansion_darker.position = (5, self.expansion_scroll_lighter.get_bottom() + 10)
        self.lbl_cc_expansion_darker.set_background(general_background)
        self.lbl_cc_expansion_darker.set_color(text_color)
        self.container_cc_expansion.append(self.lbl_cc_expansion_darker)

        self.expansion_scroll_darker = ScreenHorizontalScroll("expansion_scroll_darker", 0, 100, 0, 10)
        self.expansion_scroll_darker.position = (5, self.lbl_cc_expansion_darker.get_bottom() + 10)
        self.expansion_scroll_darker.width = container_width - 10
        self.expansion_scroll_darker.scroll_callback = self.expansion_scroll_darker_change
        self.container_cc_expansion.append(self.expansion_scroll_darker)

        self.lbl_expand_confirm = ScreenLabel("lbl_confirm", "Expand CC Pixels", 21, container_width - 10, 1)
        self.lbl_expand_confirm.position = (5, self.expansion_scroll_darker.get_bottom() + 15)
        self.lbl_expand_confirm.set_background(general_background)
        self.lbl_expand_confirm.set_color(text_color)
        self.container_cc_expansion.append(self.lbl_expand_confirm)

        self.btn_expand_accept = ScreenButton("btn_expand_accept", "Accept", 21, 130)
        self.btn_expand_accept.set_colors(button_text_color, button_back_color)
        self.btn_expand_accept.position = (10, self.lbl_expand_confirm.get_bottom() + 10)
        self.btn_expand_accept.click_callback = self.btn_expand_accept_click
        self.container_cc_expansion.append(self.btn_expand_accept)

        self.btn_expand_cancel = ScreenButton("btn_expand_cancel", "Cancel", 21, 130)
        self.btn_expand_cancel.set_colors(button_text_color, button_back_color)
        self.btn_expand_cancel.position = (container_width - self.btn_expand_cancel.width - 10,
                                           self.lbl_expand_confirm.get_bottom() + 10)
        self.btn_expand_cancel.click_callback = self.btn_expand_cancel_click
        self.container_cc_expansion.append(self.btn_expand_cancel)
        self.container_cc_expansion.visible = False

        # =======================================

        self.container_pixelwise_displace = ScreenContainer("container_pixelwise_displace", (container_width, 220),
                                                            back_color=general_background)

        self.container_pixelwise_displace.position = (self.container_zoom_buttons.get_left(), self.container_zoom_buttons.get_bottom() + 10)
        self.elements.append(self.container_pixelwise_displace)

        self.lbl_pixelwise_disp = ScreenLabel("lbl_pixelwise_disp", "Displacing Pixels", 21, container_width - 10, 1)
        self.lbl_pixelwise_disp.position = (5, 5)
        self.lbl_pixelwise_disp.set_background(general_background)
        self.lbl_pixelwise_disp.set_color(text_color)
        self.container_pixelwise_displace.append(self.lbl_pixelwise_disp)

        self.lbl_pixelwise_disp_add = ScreenLabel("lbl_pixelwise_disp_add", "Adding: 0 of 0", 21,
                                                  container_width - 10, 1)
        self.lbl_pixelwise_disp_add.position = (5, self.lbl_pixelwise_disp.get_bottom() + 10)
        self.lbl_pixelwise_disp_add.set_background(general_background)
        self.lbl_pixelwise_disp_add.set_color(text_color)
        self.container_pixelwise_displace.append(self.lbl_pixelwise_disp_add)

        self.lbl_pixelwise_disp_del = ScreenLabel("lbl_pixelwise_disp_del", "Removing: 0 of 0", 21,
                                                  container_width - 10, 1)
        self.lbl_pixelwise_disp_del.position = (5, self.lbl_pixelwise_disp_add.get_bottom() + 10)
        self.lbl_pixelwise_disp_del.set_background(general_background)
        self.lbl_pixelwise_disp_del.set_color(text_color)
        self.container_pixelwise_displace.append(self.lbl_pixelwise_disp_del)

        self.pixelwise_displacement_scroll = ScreenHorizontalScroll("pixelwise_displacement_scroll", -100, 100, 0, 10)
        self.pixelwise_displacement_scroll.position = (5, self.lbl_pixelwise_disp_del.get_bottom() + 10)
        self.pixelwise_displacement_scroll.width = container_width - 10
        self.pixelwise_displacement_scroll.scroll_callback = self.pixelwise_displacement_scroll_change
        self.container_pixelwise_displace.append(self.pixelwise_displacement_scroll)

        self.btn_pixelwise_displacement_accept = ScreenButton("btn_pixelwise_displacement_accept", "Accept", 21, 130)
        self.btn_pixelwise_displacement_accept.set_colors(button_text_color, button_back_color)
        self.btn_pixelwise_displacement_accept.position = (10, self.pixelwise_displacement_scroll.get_bottom() + 10)
        self.btn_pixelwise_displacement_accept.click_callback = self.btn_pixelwise_displacement_accept_click
        self.container_pixelwise_displace.append(self.btn_pixelwise_displacement_accept)

        self.btn_pixelwise_displacement_cancel = ScreenButton("btn_pixelwise_displacement_cancel", "Cancel", 21, 130)
        self.btn_pixelwise_displacement_cancel.set_colors(button_text_color, button_back_color)
        self.btn_pixelwise_displacement_cancel.position = (container_width - self.btn_pixelwise_displacement_cancel.width - 10,
                                                           self.pixelwise_displacement_scroll.get_bottom() + 10)
        self.btn_pixelwise_displacement_cancel.click_callback = self.btn_pixelwise_displacement_cancel_click
        self.container_pixelwise_displace.append(self.btn_pixelwise_displacement_cancel)

        self.container_pixelwise_displace.visible = False

        # =======================================

        # View panel with view control buttons
        self.container_view_buttons = ScreenContainer("container_view_buttons", (container_width, 200),
                                                      back_color=general_background)
        self.container_view_buttons.position = (self.width - container_width - 10, self.container_zoom_buttons.get_bottom() + 5)
        self.elements.append(self.container_view_buttons)

        # ===========================
        self.lbl_views = ScreenLabel("lbl_zoom", "Views", 21, button_3_width, 1)
        self.lbl_views.position = (button_3_left, 5)
        self.lbl_views.set_background(general_background)
        self.lbl_views.set_color(text_color)
        self.container_view_buttons.append(self.lbl_views)

        self.btn_view_raw = ScreenButton("btn_view_raw", "Raw", 21, button_3_width)
        self.btn_view_raw.set_colors(button_text_color, button_back_color)
        self.btn_view_raw.position = (button_3_middle, 5)
        self.btn_view_raw.click_callback = self.btn_view_raw_click
        self.container_view_buttons.append(self.btn_view_raw)

        self.btn_view_gray = ScreenButton("btn_view_gray", "Gray", 21, button_3_width)
        self.btn_view_gray.set_colors(button_text_color, button_back_color)
        self.btn_view_gray.position = (button_3_right, 5)
        self.btn_view_gray.click_callback = self.btn_view_gray_click
        self.container_view_buttons.append(self.btn_view_gray)

        self.btn_view_binary = ScreenButton("btn_view_binary", "Binary", 21, button_3_width)
        self.btn_view_binary.set_colors(button_text_color, button_back_color)
        self.btn_view_binary.position = (button_3_left, self.btn_view_gray.get_bottom() + 10)
        self.btn_view_binary.click_callback = self.btn_view_bin_click
        self.container_view_buttons.append(self.btn_view_binary)

        self.btn_view_combo_hard = ScreenButton("btn_view_combo_hard", "CC Hard", 21, button_3_width)
        self.btn_view_combo_hard.set_colors(button_text_color, button_back_color)
        self.btn_view_combo_hard.position = (button_3_middle, self.btn_view_gray.get_bottom() + 10)
        self.btn_view_combo_hard.click_callback = self.btn_view_combo_hard_click
        self.container_view_buttons.append(self.btn_view_combo_hard)

        self.btn_view_combo_soft = ScreenButton("btn_view_combo_soft", "CC Soft", 21, button_3_width)
        self.btn_view_combo_soft.set_colors(button_text_color, button_back_color)
        self.btn_view_combo_soft.position = (button_3_right, self.btn_view_gray.get_bottom() + 10)
        self.btn_view_combo_soft.click_callback = self.btn_view_combo_soft_click
        self.container_view_buttons.append(self.btn_view_combo_soft)

        # ===========================
        self.lbl_show_cc = ScreenLabel("lbl_show_cc", "Display CC", 21, button_3_width, 1)
        self.lbl_show_cc.position = (button_3_left, self.btn_view_combo_soft.get_bottom() + 20)
        self.lbl_show_cc.set_background(general_background)
        self.lbl_show_cc.set_color(text_color)
        self.container_view_buttons.append(self.lbl_show_cc)

        self.btn_show_cc_black = ScreenButton("btn_show_cc_black", "Black", 21, button_3_width)
        self.btn_show_cc_black.set_colors(button_text_color, button_back_color)
        self.btn_show_cc_black.position = (button_3_middle, self.btn_view_combo_soft.get_bottom() + 10)
        self.btn_show_cc_black.click_callback = self.btn_show_cc_black_click
        self.container_view_buttons.append(self.btn_show_cc_black)

        self.btn_show_cc_colored = ScreenButton("btn_show_cc_colored", "Colored", 21, button_3_width)
        self.btn_show_cc_colored.set_colors(button_text_color, button_back_color)
        self.btn_show_cc_colored.position = (button_3_right, self.btn_view_combo_soft.get_bottom() + 10)
        self.btn_show_cc_colored.click_callback = self.btn_show_cc_colored_click
        self.container_view_buttons.append(self.btn_show_cc_colored)

        # ===========================
        self.lbl_small_highlight = ScreenLabel("lbl_small_highlight", "Highlight CCs smaller than: " + str(self.min_highlight_size), 21, 290, 1)
        self.lbl_small_highlight.position = (5, self.btn_show_cc_black.get_bottom() + 20)
        self.lbl_small_highlight.set_background(general_background)
        self.lbl_small_highlight.set_color(text_color)
        self.container_view_buttons.append(self.lbl_small_highlight)

        self.highlight_scroll = ScreenHorizontalScroll("highlight_scroll", 0, 100, 0, 10)
        self.highlight_scroll.position = (5, self.lbl_small_highlight.get_bottom() + 10)
        self.highlight_scroll.width = container_width - 10
        self.highlight_scroll.scroll_callback = self.highlight_scroll_change
        self.container_view_buttons.append(self.highlight_scroll)

        # ===========================
        self.container_edition_mode = ScreenContainer("container_edition_mode", (container_width, 140),
                                                      back_color=general_background)
        self.container_edition_mode.position = (self.width - container_width - 10, self.container_view_buttons.get_bottom() + 5)
        self.elements.append(self.container_edition_mode)

        # self.btn_edition_start = ScreenButton("btn_edition_start", "Edit Pixels", 21, button_3_width)
        # self.btn_edition_start.position = (button_3_left, 5)
        self.btn_edition_start = ScreenButton("btn_edition_start", "Edit [P]ixels", 21, button_2_width)
        self.btn_edition_start.position = (button_2_left, 5)
        self.btn_edition_start.set_colors(button_text_color, button_back_color)
        self.btn_edition_start.click_callback = self.btn_edition_start_click
        self.container_edition_mode.append(self.btn_edition_start)

        self.btn_edition_delete = ScreenButton("btn_edition_delete", "[D]elete CC", 21, button_2_width)
        self.btn_edition_delete.position = (button_2_right, 5)
        self.btn_edition_delete.set_colors(button_text_color, button_back_color)
        self.btn_edition_delete.click_callback = self.btn_edition_delete_click
        self.container_edition_mode.append(self.btn_edition_delete)

        self.btn_edition_expand = ScreenButton("btn_edition_expand", "[E]xpand CC", 21, button_2_width)
        self.btn_edition_expand.position = (button_2_left, self.btn_edition_delete.get_bottom() + 10)
        self.btn_edition_expand.set_colors(button_text_color, button_back_color)
        self.btn_edition_expand.click_callback = self.btn_edition_expand_click
        self.container_edition_mode.append(self.btn_edition_expand)

        self.btn_edition_shrink = ScreenButton("btn_edition_shrink", "[S]hrink CC", 21, button_2_width)
        self.btn_edition_shrink.position = (button_2_right, self.btn_edition_delete.get_bottom() + 10)
        self.btn_edition_shrink.set_colors(button_text_color, button_back_color)
        self.btn_edition_shrink.click_callback = self.btn_edition_shrink_click
        self.container_edition_mode.append(self.btn_edition_shrink)

        self.btn_displacement_hor = ScreenButton("btn_displacement_hor", "H. Displace", 21, button_2_width)
        self.btn_displacement_hor.position = (button_2_left, self.btn_edition_expand.get_bottom() + 10)
        self.btn_displacement_hor.set_colors(button_text_color, button_back_color)
        self.btn_displacement_hor.click_callback = self.btn_displacement_hor_click
        self.container_edition_mode.append(self.btn_displacement_hor)

        self.btn_displacement_ver = ScreenButton("btn_displacement_ver", "V. Displace", 21, button_2_width)
        self.btn_displacement_ver.position = (button_2_right, self.btn_edition_shrink.get_bottom() + 10)
        self.btn_displacement_ver.set_colors(button_text_color, button_back_color)
        self.btn_displacement_ver.click_callback = self.btn_displacement_ver_click
        self.container_edition_mode.append(self.btn_displacement_ver)

        # ===========================
        self.container_pixels_edit = ScreenContainer("container_pixels_edit", (container_width, 100),
                                                     back_color=general_background)
        self.container_pixels_edit.position = (self.width - container_width - 10, self.container_view_buttons.get_bottom() + 5)
        self.elements.append(self.container_pixels_edit)

        self.btn_edition_stop = ScreenButton("btn_edition_stop", "Stop Editing Pixels", 21, button_width)
        self.btn_edition_stop.position = (button_left, 5)
        self.btn_edition_stop.set_colors(button_text_color, button_back_color)
        self.btn_edition_stop.click_callback = self.btn_edition_stop_click
        self.container_pixels_edit.append(self.btn_edition_stop)

        self.container_pixels_edit.visible = False


        # ===========================
        # Panel with confirmation buttons (Message, Accept, Cancel)
        self.container_confirm_buttons = ScreenContainer("container_confirm_buttons", (300, 70), back_color=general_background)
        self.container_confirm_buttons.position = (self.container_view_buttons.get_left(), self.container_edition_mode.get_bottom() + 10)
        self.elements.append(self.container_confirm_buttons)

        self.lbl_confirm = ScreenLabel("lbl_confirm", "Exit without saving?", 21, 290, 1)
        self.lbl_confirm.position = (5, 5)
        self.lbl_confirm.set_background(general_background)
        self.lbl_confirm.set_color(text_color)
        self.container_confirm_buttons.append(self.lbl_confirm)

        self.btn_confirm_accept = ScreenButton("btn_confirm_accept", "Accept", 21, 130)
        self.btn_confirm_accept.set_colors(button_text_color, button_back_color)
        self.btn_confirm_accept.position = (10, self.lbl_confirm.get_bottom() + 10)
        self.btn_confirm_accept.click_callback = self.btn_confirm_accept_click
        self.container_confirm_buttons.append(self.btn_confirm_accept)

        self.btn_confirm_cancel = ScreenButton("btn_confirm_cancel", "Cancel", 21, 130)
        self.btn_confirm_cancel.set_colors(button_text_color, button_back_color)
        self.btn_confirm_cancel.position = (self.container_confirm_buttons.width - self.btn_confirm_cancel.width - 10,
                                            self.lbl_confirm.get_bottom() + 10)
        self.btn_confirm_cancel.click_callback = self.btn_confirm_cancel_click
        self.container_confirm_buttons.append(self.btn_confirm_cancel)
        self.container_confirm_buttons.visible = False

        # =============================
        stats_background = (60, 20, 10)
        self.container_stats = ScreenContainer("container_stats", (container_width, 160), back_color=stats_background)
        self.container_stats.position = (self.width - container_width - 10, self.container_edition_mode.get_bottom() + 5)
        self.elements.append(self.container_stats)

        self.lbl_pixel_stats = ScreenLabel("lbl_pixel_stats", "Pixel Stats", 21, container_width - 10, 1)
        self.lbl_pixel_stats.position = (5, 5)
        self.lbl_pixel_stats.set_background(stats_background)
        self.lbl_pixel_stats.set_color(text_color)
        self.container_stats.append(self.lbl_pixel_stats)

        self.lbl_pixels_white = ScreenLabel("lbl_pixels_white", "White:\n1000000", 21, button_3_width, 1)
        self.lbl_pixels_white.position = (button_3_left, self.lbl_pixel_stats.get_bottom() + 10)
        self.lbl_pixels_white.set_background(stats_background)
        self.lbl_pixels_white.set_color(text_color)
        self.container_stats.append(self.lbl_pixels_white)

        self.lbl_pixels_black = ScreenLabel("lbl_pixels_black", "Black:\n1000000", 21, button_3_width, 1)
        self.lbl_pixels_black.position = (button_3_middle, self.lbl_pixel_stats.get_bottom() + 10)
        self.lbl_pixels_black.set_background(stats_background)
        self.lbl_pixels_black.set_color(text_color)
        self.container_stats.append(self.lbl_pixels_black)

        self.lbl_pixels_ratio = ScreenLabel("lbl_pixels_ratio", "Ratio:\n0.00000", 21, button_3_width, 1)
        self.lbl_pixels_ratio.position = (button_3_right, self.lbl_pixel_stats.get_bottom() + 10)
        self.lbl_pixels_ratio.set_background(stats_background)
        self.lbl_pixels_ratio.set_color(text_color)
        self.container_stats.append(self.lbl_pixels_ratio)

        self.lbl_cc_stats = ScreenLabel("lbl_cc_stats", "CC Stats", 21, container_width - 10, 1)
        self.lbl_cc_stats.position = (5, self.lbl_pixels_ratio.get_bottom() + 10)
        self.lbl_cc_stats.set_background(stats_background)
        self.lbl_cc_stats.set_color(text_color)
        self.container_stats.append(self.lbl_cc_stats)

        self.lbl_cc_stats_count = ScreenLabel("lbl_cc_stats_count", "Total CC: 0", 21, container_width - 10, 1)
        self.lbl_cc_stats_count.position = (5, self.lbl_cc_stats.get_bottom() + 10)
        self.lbl_cc_stats_count.set_background(stats_background)
        self.lbl_cc_stats_count.set_color(text_color)
        self.container_stats.append(self.lbl_cc_stats_count)

        self.lbl_cc_stats_min = ScreenLabel("lbl_cc_stats_min", "Min:\n0000", 21, button_4_width, 1)
        self.lbl_cc_stats_min.position = (button_4_left_1, self.lbl_cc_stats_count.get_bottom() + 10)
        self.lbl_cc_stats_min.set_background(stats_background)
        self.lbl_cc_stats_min.set_color(text_color)
        self.container_stats.append(self.lbl_cc_stats_min)

        self.lbl_cc_stats_max = ScreenLabel("lbl_cc_stats_max", "Max:\n0000", 21, button_4_width, 1)
        self.lbl_cc_stats_max.position = (button_4_left_2, self.lbl_cc_stats_count.get_bottom() + 10)
        self.lbl_cc_stats_max.set_background(stats_background)
        self.lbl_cc_stats_max.set_color(text_color)
        self.container_stats.append(self.lbl_cc_stats_max)

        self.lbl_cc_stats_mean = ScreenLabel("lbl_cc_stats_mean", "Mean:\n0000", 21, button_4_width, 1)
        self.lbl_cc_stats_mean.position = (button_4_left_3, self.lbl_cc_stats_count.get_bottom() + 10)
        self.lbl_cc_stats_mean.set_background(stats_background)
        self.lbl_cc_stats_mean.set_color(text_color)
        self.container_stats.append(self.lbl_cc_stats_mean)

        self.lbl_cc_stats_median = ScreenLabel("lbl_cc_stats_median", "Median:\n0000", 21, button_4_width, 1)
        self.lbl_cc_stats_median.position = (button_4_left_4, self.lbl_cc_stats_count.get_bottom() + 10)
        self.lbl_cc_stats_median.set_background(stats_background)
        self.lbl_cc_stats_median.set_color(text_color)
        self.container_stats.append(self.lbl_cc_stats_median)

        # ============================================================================
        # Panel with state buttons (Undo, Redo, return accept, return cancel)
        self.container_state_buttons = ScreenContainer("container_state_buttons", (container_width, 90), general_background)
        self.container_state_buttons.position = (self.container_stats.get_left(), self.container_stats.get_bottom() + 10)
        self.elements.append(self.container_state_buttons)

        self.btn_undo = ScreenButton("btn_undo", "Undo", 21, button_2_width)
        self.btn_undo.set_colors(button_text_color, button_back_color)
        self.btn_undo.position = (button_2_left, 5)
        self.btn_undo.click_callback = self.btn_undo_click
        self.container_state_buttons.append(self.btn_undo)

        self.btn_redo = ScreenButton("btn_redo", "Redo", 21, button_2_width)
        self.btn_redo.set_colors(button_text_color, button_back_color)
        self.btn_redo.position = (button_2_right, 5)
        self.btn_redo.click_callback = self.btn_redo_click
        self.container_state_buttons.append(self.btn_redo)

        # Secondary screen mode
        # Add Cancel Button
        self.btn_return_cancel = ScreenButton("btn_return_cancel", "Cancel", 21, button_2_width)
        self.btn_return_cancel.set_colors(button_text_color, button_back_color)
        self.btn_return_cancel.position = (button_2_left, self.btn_redo.get_bottom() + 15)
        self.btn_return_cancel.click_callback = self.btn_return_cancel_click
        self.container_state_buttons.append(self.btn_return_cancel)

        # Add Accept Button
        self.btn_return_accept = ScreenButton("btn_return_accept", "Accept", 21, button_2_width)
        self.btn_return_accept.set_colors(button_text_color, button_back_color)
        self.btn_return_accept.position = (button_2_right, self.btn_redo.get_bottom() + 15)
        self.btn_return_accept.click_callback = self.btn_return_accept_click
        self.container_state_buttons.append(self.btn_return_accept)

        # ============================================================================
        image_width = self.width - self.container_view_buttons.width - 30
        image_height = self.height - 20
        self.container_images = ScreenContainer("container_images", (image_width, image_height), back_color=(0, 0, 0))
        self.container_images.position = (10, 10)
        self.elements.append(self.container_images)

        # ... image objects ...
        tempo_blank = np.zeros((50, 50, 3), np.uint8)
        tempo_blank[:, :, :] = 255
        self.img_main = ScreenImage("img_raw", tempo_blank, 0, 0, True, cv2.INTER_NEAREST)
        self.img_main.position = (0, 0)
        self.img_main.mouse_button_down_callback = self.img_mouse_down
        self.img_main.double_click_callback = self.img_double_click
        self.img_main.mouse_motion_callback = self.img_mouse_motion
        self.container_images.append(self.img_main)

        self.pre_edition_binary = None
        self.undo_stack = []
        self.redo_stack = []

        self.finished_callback = None
        self.parent_screen = parent_screen

        self.last_motion_set_x = None
        self.last_motion_set_y = None
        self.last_motion_polarity = None

        self.last_plus_minus_time = None

        self.elements.key_up_callback = self.main_key_up

        self.update_cc_info()
        self.update_current_view(True)

    def update_current_view(self, resized=False, region=None):
        if region is None:
            if self.view_mode == GTPixelBinaryAnnotator.ViewModeGray:
                base_image = self.base_gray
            elif self.view_mode == GTPixelBinaryAnnotator.ViewModeBinary:
                if self.cc_show_mode == GTPixelBinaryAnnotator.CCShowBlack:
                    base_image = np.zeros((self.base_binary.shape[0], self.base_binary.shape[1], 3), dtype=np.uint8)

                    base_image[:, :, 0] = self.base_binary
                    base_image[:, :, 1] = self.base_binary
                    base_image[:, :, 2] = self.base_binary
                else:
                    base_image = self.base_cc_image.copy()
                    base_image[self.base_binary > 0, :] = 255

            elif self.view_mode == GTPixelBinaryAnnotator.ViewModeHardCombined:
                base_image = self.base_gray.copy()

                # create hard combined
                inverse_binary_mask = self.base_binary == 0
                if self.cc_show_mode == GTPixelBinaryAnnotator.CCShowBlack:
                    base_image[inverse_binary_mask, :] = 0
                else:
                    base_image[inverse_binary_mask, :] = self.base_cc_image[inverse_binary_mask, :].copy()

            elif self.view_mode == GTPixelBinaryAnnotator.ViewModeSoftCombined:
                if self.cc_show_mode == GTPixelBinaryAnnotator.CCShowBlack:
                    base_image = np.zeros((self.base_binary.shape[0], self.base_binary.shape[1], 3), dtype=np.uint8)

                    scaled_black = 1 - (self.base_binary / 255)
                    base_image[:, :, 0] = scaled_black
                else:
                    base_image = self.base_cc_image.copy()


                # replace the empty channels on each CC with original CC grayscale
                for channel in range(3):
                    inverse_binary_mask = base_image[:, :, channel] == 0

                    base_image[inverse_binary_mask, channel] = self.base_gray[inverse_binary_mask, 0].copy()
            else:
                base_image = self.base_raw.copy()
        else:
            start_x = max(region[0], 0)
            end_x = min(region[1] + 1, self.base_raw.shape[1])
            start_y = max(region[2], 0)
            end_y = min(region[3] + 1, self.base_raw.shape[0])

            # create empty image
            base_image = np.zeros((self.base_binary.shape[0], self.base_binary.shape[1], 3), dtype=np.uint8)

            if self.view_mode == GTPixelBinaryAnnotator.ViewModeGray:
                base_image[start_y:end_y, start_x:end_x] = self.base_gray[start_y:end_y, start_x:end_x].copy()

            elif self.view_mode == GTPixelBinaryAnnotator.ViewModeBinary:
                if self.cc_show_mode == GTPixelBinaryAnnotator.CCShowBlack:

                    base_image[start_y:end_y, start_x:end_x, 0] = self.base_binary[start_y:end_y, start_x:end_x].copy()
                    base_image[start_y:end_y, start_x:end_x, 1] = self.base_binary[start_y:end_y, start_x:end_x].copy()
                    base_image[start_y:end_y, start_x:end_x, 2] = self.base_binary[start_y:end_y, start_x:end_x].copy()
                else:
                    base_image[start_y:end_y, start_x:end_x] = self.base_cc_image[start_y:end_y, start_x:end_x].copy()

                    cut_mask = self.base_binary[start_y:end_y, start_x:end_x] > 0
                    tempo_cut = base_image[start_y:end_y, start_x:end_x]
                    tempo_cut[cut_mask, :] = 255

            elif self.view_mode == GTPixelBinaryAnnotator.ViewModeHardCombined:
                base_image[start_y:end_y, start_x:end_x] = self.base_gray[start_y:end_y, start_x:end_x].copy()

                # create hard combined
                inverse_binary_mask = self.base_binary[start_y:end_y, start_x:end_x] == 0
                tempo_cut = base_image[start_y:end_y, start_x:end_x]

                if self.cc_show_mode == GTPixelBinaryAnnotator.CCShowBlack:
                    tempo_cut[inverse_binary_mask, :] = 0
                else:
                    cc_image_cut = self.base_cc_image[start_y:end_y, start_x:end_x].copy()
                    tempo_cut[inverse_binary_mask, :] = cc_image_cut[inverse_binary_mask, :]

            elif self.view_mode == GTPixelBinaryAnnotator.ViewModeSoftCombined:
                if self.cc_show_mode == GTPixelBinaryAnnotator.CCShowBlack:

                    scaled_black = 1 - (self.base_binary[start_y:end_y, start_x:end_x] / 255)
                    base_image[start_y:end_y, start_x:end_x, 0] = scaled_black
                else:
                    base_image[start_y:end_y, start_x:end_x] = self.base_cc_image[start_y:end_y, start_x:end_x].copy()

                # replace the empty channels on each CC with original CC grayscale
                base_cut = base_image[start_y:end_y, start_x:end_x, :]
                gray_cut = self.base_gray[start_y:end_y, start_x:end_x, 0].copy()
                for channel in range(3):
                    inverse_binary_mask = base_image[start_y:end_y, start_x:end_x, channel] == 0

                    base_cut[inverse_binary_mask, channel] = gray_cut[inverse_binary_mask].copy()
                    #base_image[inverse_binary_mask, channel] = self.base_gray[inverse_binary_mask, 0].copy()
            else:
                base_image[start_y:end_y, start_x:end_x] = self.base_raw[start_y:end_y, start_x:end_x].copy()

        h, w, c = base_image.shape

        modified_image = base_image.copy()

        if self.selected_CC is not None:
            # mark selected CC ...
            cc = self.selected_CC
            current_cut = modified_image[cc.min_y:cc.max_y + 1, cc.min_x:cc.max_x + 1]
            cc_mask = cc.img > 0
            current_cut[cc_mask, 0] = 255

            # Highlight pixels for current threshold
            if self.editor_mode == GTPixelBinaryAnnotator.ModeGrowCC_Growing:
                for value, y, x in self.CC_expansion_pixels_light[:self.count_expand_light]:
                    modified_image[y, x, 1] = 255
                for value, y, x in self.CC_expansion_pixels_dark[:self.count_expand_dark]:
                    modified_image[y, x, 1] = 255

            elif self.editor_mode == GTPixelBinaryAnnotator.ModeShrinkCC_Shrinking:
                for value, y, x in self.CC_shrinking_pixels_light[:self.count_shrink_light]:
                    modified_image[y, x, 1] = 255
                for value, y, x in self.CC_shrinking_pixels_dark[:self.count_shrink_dark]:
                    modified_image[y, x, 1] = 255

        if self.editor_mode == GTPixelBinaryAnnotator.ModePixelwise_Displacing:
            # highlight pixels for current threshold ...
            for value, y, x in self.displacement_pixels_adding:
                modified_image[y, x, 1] = 255

            for value, y, x in self.displacement_pixels_removing:
                modified_image[y, x, 0] = 255

        # show highlighted small CC (if any)
        if self.min_highlight_size > 0:
            for cc in self.base_ccs:
                if cc.size < self.min_highlight_size:
                    # print(str((cc.getCenter(), cc.size, cc.min_x, cc.max_x, cc.min_y, cc.max_y)))
                    # compute highlight base radius
                    base_radius = math.sqrt(math.pow(cc.getWidth() / 2, 2.0) + math.pow(cc.getHeight() / 2, 2.0))
                    highlight_radius = int(base_radius * 3)

                    cc_cx, cc_cy = cc.getCenter()
                    cv2.circle(modified_image, (int(cc_cx), int(cc_cy)), highlight_radius, (255, 0, 0), 2)

        if region is None:
            # resize ...
            new_res = (int(w * self.view_scale), int(h * self.view_scale))
            modified_image = cv2.resize(modified_image, new_res, interpolation=cv2.INTER_NEAREST)

            # add grid
            if self.view_scale >= 4.0:
                int_scale = int(self.view_scale)
                density = 2
                modified_image[int_scale - 1::int_scale, ::density, :] = 128
                modified_image[::density, int_scale - 1::int_scale, :] = 128

            # replace/update image
            self.img_main.set_image(modified_image, 0, 0, True, cv2.INTER_NEAREST)
        else:
            new_res = (int((end_x - start_x) * self.view_scale), int((end_y - start_y) * self.view_scale))
            portion_cut = cv2.resize(modified_image[start_y:end_y, start_x:end_x, :], new_res, interpolation=cv2.INTER_NEAREST)

            # add grid
            if self.view_scale >= 4.0:
                int_scale = int(self.view_scale)
                density = 2

                old_off = int_scale - 1
                new_off_x = int(math.ceil((start_x * self.view_scale - old_off) / int_scale) * int_scale + old_off - start_x * self.view_scale)
                new_off_y = int(math.ceil((start_y * self.view_scale - old_off) / int_scale) * int_scale + old_off - start_y * self.view_scale)
                portion_cut[new_off_y::int_scale, ::density, :] = 128
                portion_cut[::density, new_off_x::int_scale, :] = 128

            # update region ....
            self.img_main.update_image_region(portion_cut, (int(start_x * self.view_scale), int(start_y * self.view_scale)))

        if resized:
            self.container_images.recalculate_size()

    def update_cc_info(self):
        h, w, _ = self.base_raw.shape

        fake_age = np.zeros((h, w), dtype=np.float32)
        self.base_ccs = Labeler.extractSpatioTemporalContent((255 - self.base_binary), fake_age, False)

        self.base_cc_image = np.zeros((h, w, 3), dtype=np.uint8)

        tempo_sizes = []
        for idx, cc in enumerate(self.base_ccs):
            n_colors = len(GTPixelBinaryAnnotator.CCShowColors)
            cc_color = GTPixelBinaryAnnotator.CCShowColors[idx % n_colors]

            current_cut = self.base_cc_image[cc.min_y:cc.max_y + 1, cc.min_x:cc.max_x + 1]
            cc_mask = cc.img > 0

            current_cut[cc_mask, 0] += cc_color[0]
            current_cut[cc_mask, 1] += cc_color[1]
            current_cut[cc_mask, 2] += cc_color[2]

            tempo_sizes.append(cc.size)

        # self.base_cc_mask = np.sum(self.base_cc_image, 2) == 0
        total_pixels = h * w
        self.stats_pixels_white = self.base_binary.sum() / 255
        self.stats_pixels_black = total_pixels - self.stats_pixels_white
        self.stats_pixels_ratio = self.stats_pixels_black / total_pixels

        cc_sizes = np.array(tempo_sizes)
        self.stats_cc_count = len(self.base_ccs)
        self.stats_cc_size_min = cc_sizes.min()
        self.stats_cc_size_max = cc_sizes.max()
        self.stats_cc_size_mean = cc_sizes.mean()
        self.stats_cc_size_median = np.median(cc_sizes)

        # update interface ...
        self.lbl_pixels_white.set_text("White:\n" + str(self.stats_pixels_white))
        self.lbl_pixels_black.set_text("Black:\n" + str(self.stats_pixels_black))
        self.lbl_pixels_ratio.set_text("Ratio:\n{0:.5f}".format(self.stats_pixels_ratio))

        self.lbl_cc_stats_count.set_text("Total CC: {0}".format(self.stats_cc_count))
        self.lbl_cc_stats_min.set_text("Min:\n{0}".format(self.stats_cc_size_min))
        self.lbl_cc_stats_max.set_text("Max:\n{0}".format(self.stats_cc_size_max))
        self.lbl_cc_stats_mean.set_text("Mean:\n{0:.2f}".format(self.stats_cc_size_mean))
        self.lbl_cc_stats_median.set_text("Median:\n{0:.2f}".format(self.stats_cc_size_median))

    def update_view_scale(self, new_scale):
        prev_scale = self.view_scale

        if (new_scale < 0.25) or (self.max_scale is not None and new_scale > self.max_scale):
            # below minimum or above maximum
            return

        self.view_scale = new_scale

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
        try:
            self.update_current_view(True)
        except:
            # rescale failed ...
            # restore previous scale
            self.view_scale = prev_scale
            self.max_scale = prev_scale
            self.update_current_view(True)
            print("Maximum zoom is {0:.2f}%".format(self.max_scale * 100))


        # set offsets
        if self.container_images.v_scroll.active and 0 <= new_off_y <= self.container_images.v_scroll.max:
            self.container_images.v_scroll.value = new_off_y
        if self.container_images.h_scroll.active and 0 <= new_off_x <= self.container_images.h_scroll.max:
            self.container_images.h_scroll.value = new_off_x

        # update scale text ...
        self.lbl_zoom.set_text("Zoom: " + str(int(round(self.view_scale * 100,0))) + "%")

    def btn_zoom_reduce_click(self, button):
        if self.view_scale <= 1.0:
            self.update_view_scale(self.view_scale - 0.25)
        else:
            self.update_view_scale(self.view_scale - 1.0)

    def btn_zoom_increase_click(self, button):
        if self.view_scale < 1.0:
            self.update_view_scale(self.view_scale + 0.25)
        else:
            self.update_view_scale(self.view_scale + 1.0)

    def btn_zoom_zero_click(self, button):
        self.update_view_scale(1.0)

    def btn_view_raw_click(self, button):
        self.view_mode = GTPixelBinaryAnnotator.ViewModeRaw
        self.update_current_view()

    def btn_view_gray_click(self, button):
        self.view_mode = GTPixelBinaryAnnotator.ViewModeGray
        self.update_current_view()

    def btn_view_bin_click(self, button):
        self.view_mode = GTPixelBinaryAnnotator.ViewModeBinary
        self.update_current_view()

    def btn_view_combo_soft_click(self, button):
        self.view_mode = GTPixelBinaryAnnotator.ViewModeSoftCombined
        self.update_current_view()

    def btn_view_combo_hard_click(self, button):
        self.view_mode = GTPixelBinaryAnnotator.ViewModeHardCombined
        self.update_current_view()

    def btn_show_cc_black_click(self, button):
        self.cc_show_mode = GTPixelBinaryAnnotator.CCShowBlack
        self.update_current_view()

    def btn_show_cc_colored_click(self, button):
        self.cc_show_mode = GTPixelBinaryAnnotator.CCShowColored
        self.update_current_view()

    def highlight_scroll_change(self, scroll):
        self.min_highlight_size = int(scroll.value)
        self.lbl_small_highlight.set_text("Highlight CCs smaller than: " + str(self.min_highlight_size))

        self.update_current_view(False)

    def set_editor_mode(self, new_mode):
        self.editor_mode = new_mode

        self.container_confirm_buttons.visible = (new_mode == GTPixelBinaryAnnotator.ModeConfirmCancel or
                                                  new_mode == GTPixelBinaryAnnotator.ModeDeleteCC_Select)

        if new_mode == GTPixelBinaryAnnotator.ModeConfirmCancel:
            self.lbl_confirm.set_text("Exit without saving?")
            self.btn_confirm_accept.visible = True
        elif new_mode == GTPixelBinaryAnnotator.ModeDeleteCC_Select:
            self.lbl_confirm.set_text("Select a CC to Delete")
            self.btn_confirm_accept.visible = False

        self.container_state_buttons.visible = (new_mode == GTPixelBinaryAnnotator.ModeNavigation)

        self.container_stats.visible = (new_mode == GTPixelBinaryAnnotator.ModeNavigation)

        self.container_edition_mode.visible = new_mode == GTPixelBinaryAnnotator.ModeNavigation

        self.container_pixels_edit.visible = new_mode == GTPixelBinaryAnnotator.ModeEdition

        self.container_view_buttons.visible = (new_mode != GTPixelBinaryAnnotator.ModeGrowCC_Growing and
                                               new_mode != GTPixelBinaryAnnotator.ModeShrinkCC_Shrinking and
                                               new_mode != GTPixelBinaryAnnotator.ModePixelwise_Displacing)

        self.container_cc_expansion.visible = (new_mode == GTPixelBinaryAnnotator.ModeGrowCC_Growing or
                                               new_mode == GTPixelBinaryAnnotator.ModeShrinkCC_Shrinking)

        self.container_pixelwise_displace.visible = new_mode == GTPixelBinaryAnnotator.ModePixelwise_Displacing

        if self.container_cc_expansion.visible:
            self.update_label_expansion()


    def btn_edition_start_click(self, button):
        self.pre_edition_binary = self.base_binary.copy()

        self.set_editor_mode(GTPixelBinaryAnnotator.ModeEdition)

    def btn_edition_stop_click(self, button):
        self.undo_stack.append({
            "operation": "pixels_edited",
            "prev_state": self.pre_edition_binary.copy(),
            "new_state": self.base_binary.copy(),
        })

        self.update_cc_info()
        self.update_current_view(False)
        self.set_editor_mode(GTPixelBinaryAnnotator.ModeNavigation)

    def btn_undo_click(self, button):
        if len(self.undo_stack) == 0:
            print("No operations to undo")
            return

        # copy last operation
        to_undo = self.undo_stack[-1]

        success = False

        if to_undo["operation"] == "pixels_edited":
            # restore previous state
            self.base_binary = to_undo["prev_state"].copy()
            success = True

        # removing ...
        if success:
            self.redo_stack.append(to_undo)
            del self.undo_stack[-1]

            # update interface ...
            self.update_cc_info()
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

        if to_redo["operation"] == "pixels_edited":
            # restore new state
            self.base_binary = to_redo["new_state"].copy()
            success = True

        # removing ...
        if success:
            self.undo_stack.append(to_redo)
            del self.redo_stack[-1]

            # update interface ...
            self.update_cc_info()
            self.update_current_view(False)
        else:
            print("Action could not be re-done")

    def btn_return_accept_click(self, button):
        if self.finished_callback is not None:
            # accept ...
            self.finished_callback(True, self.base_binary)

        self.return_screen = self.parent_screen

    def btn_return_cancel_click(self, button):
        if len(self.undo_stack) == 0:
            if self.finished_callback is not None:
                # accept ...
                self.finished_callback(False, None)

            self.return_screen = self.parent_screen
        else:
            self.set_editor_mode(GTPixelBinaryAnnotator.ModeConfirmCancel)


    def img_mouse_down(self, img_object, pos, button):
        # ... first, get click location on original image space
        scaled_x, scaled_y = pos
        click_x = int(scaled_x / self.view_scale)
        click_y = int(scaled_y / self.view_scale)

        if click_x < 0 or click_y < 0 or click_x >= self.base_raw.shape[1] or click_y >= self.base_raw.shape[0]:
            # out of boundaries
            return

        if button == 1 and self.editor_mode == GTPixelBinaryAnnotator.ModeEdition:
            # invert pixel ...
            self.base_binary[click_y, click_x] = 255 - self.base_binary[click_y, click_x]

            if self.base_binary[click_y, click_x] > 0:
                self.base_cc_image[click_y, click_x, :] = 0
            else:
                self.base_cc_image[click_y, click_x, 0] = 1
                self.base_cc_image[click_y, click_x, 1] = 0
                self.base_cc_image[click_y, click_x, 2] = 0

            self.last_motion_set_x = click_x
            self.last_motion_set_y = click_y
            self.last_motion_polarity = self.base_binary[click_y, click_x]

            self.update_current_view(False, (click_x - 1, click_x + 1, click_y - 1, click_y + 1))

        elif button == 1 and self.editor_mode == GTPixelBinaryAnnotator.ModeGrowCC_Select:
            for cc in self.base_ccs:
                if cc.min_x <= click_x <= cc.max_x and cc.min_y <= click_y <= cc.max_y:
                    rel_offset_x = click_x - cc.min_x
                    rel_offset_y = click_y - cc.min_y

                    if cc.img[rel_offset_y, rel_offset_x] > 0:
                        # cv2.imshow("selected", cc.img)
                        # select CC and move to next mode
                        self.pre_edition_binary = self.base_binary.copy()
                        self.selected_CC = cc

                        # compute CC stats ...
                        cc_cut = self.base_gray[cc.min_y:cc.max_y + 1, cc.min_x:cc.max_x + 1, 0].copy()
                        cc_mask = cc.img > 0
                        avg_cc_luminosity = cc_cut[cc_mask].mean()

                        # compute expansion
                        exp_min_x = min(GTPixelBinaryAnnotator.CCExpansionDistance, cc.min_x)
                        exp_max_x = min(GTPixelBinaryAnnotator.CCExpansionDistance, self.base_raw.shape[1] - 1 - cc.max_x)
                        exp_min_y = min(GTPixelBinaryAnnotator.CCExpansionDistance, cc.min_y)
                        exp_max_y = min(GTPixelBinaryAnnotator.CCExpansionDistance, self.base_raw.shape[0] - 1 - cc.max_y)

                        exp_h = cc.getHeight() + exp_min_y + exp_max_y
                        exp_w = cc.getWidth() + exp_min_x + exp_max_x

                        expansion = np.zeros((exp_h, exp_w), np.uint8)
                        expansion[exp_min_y:exp_h - exp_max_y, exp_min_x:exp_w - exp_max_x] = cc.img.copy()

                        dil_kernel = np.ones((1 + exp_min_y + exp_max_y, 1 + exp_min_x + exp_max_x), dtype=np.uint8)
                        dilated = cv2.dilate(expansion, np.array(dil_kernel, dtype=np.uint8))
                        expansion = dilated - expansion

                        abs_exp_min_x = cc.min_x - exp_min_x
                        abs_exp_max_x = cc.max_x + exp_max_x + 1
                        abs_exp_min_y = cc.min_y - exp_min_y
                        abs_exp_max_y = cc.max_y + exp_max_y + 1

                        # get the distance between the average luminosity of the pixel and the expansion pixels
                        expanded_cut = self.base_gray[abs_exp_min_y:abs_exp_max_y, abs_exp_min_x:abs_exp_max_x, 0].copy()
                        expanded_cut = expanded_cut.astype(np.float64) - avg_cc_luminosity

                        # find expansion pixels ....
                        lighter_expansion_pixels = []
                        darker_expansion_pixels = []
                        for y in range(expanded_cut.shape[0]):
                            for x in range(expanded_cut.shape[1]):
                                # check if part of the expansion
                                if expansion[y, x] > 0:
                                    # check if lighter or darker ....
                                    if expanded_cut[y, x] > 0:
                                        lighter_expansion_pixels.append((expanded_cut[y, x], abs_exp_min_y + y, abs_exp_min_x + x))
                                    else:
                                        darker_expansion_pixels.append((-expanded_cut[y, x], abs_exp_min_y + y, abs_exp_min_x + x))

                        self.CC_expansion_pixels_light = sorted(lighter_expansion_pixels)
                        self.CC_expansion_pixels_dark = sorted(darker_expansion_pixels)
                        self.count_expand_light = 0
                        self.count_expand_dark = 0

                        self.expansion_scroll_lighter.reset(0, len(self.CC_expansion_pixels_light), 0, 1)
                        self.expansion_scroll_darker.reset(0, len(self.CC_expansion_pixels_dark), 0, 1)

                        self.set_editor_mode(GTPixelBinaryAnnotator.ModeGrowCC_Growing)
                        # force one view
                        self.view_mode = GTPixelBinaryAnnotator.ViewModeSoftCombined
                        self.cc_show_mode = GTPixelBinaryAnnotator.CCShowBlack
                        self.update_current_view()
                        break

        elif button == 1 and self.editor_mode == GTPixelBinaryAnnotator.ModeShrinkCC_Select:
            for cc in self.base_ccs:
                if cc.min_x <= click_x <= cc.max_x and cc.min_y <= click_y <= cc.max_y:
                    rel_offset_x = click_x - cc.min_x
                    rel_offset_y = click_y - cc.min_y

                    if cc.img[rel_offset_y, rel_offset_x] > 0:
                        self.selected_CC = cc
                        self.pre_edition_binary = self.base_binary.copy()

                        # compute CC stats ...
                        cc_cut = self.base_gray[cc.min_y:cc.max_y + 1, cc.min_x:cc.max_x + 1, 0].copy()
                        cc_mask = cc.img > 0
                        avg_cc_luminosity = cc_cut[cc_mask].mean()

                        erode_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
                        exp_img = np.zeros((cc.img.shape[0] + 2, cc.img.shape[1] + 2),  dtype=np.uint8)
                        exp_img[1:1 + cc.img.shape[0], 1:1 + cc.img.shape[1]] = cc.img.copy()
                        shrunken = cv2.erode(exp_img, erode_kernel)
                        reduction = exp_img - shrunken
                        reduction = reduction[1:1 + cc.img.shape[0], 1:1 + cc.img.shape[1]]

                        # find pixels that get erased ...
                        reduction_pixels_lighter = []
                        reduction_pixels_darker = []
                        for y in range(cc_cut.shape[0]):
                            for x in range(cc_cut.shape[1]):
                                if reduction[y, x] > 0:
                                    # pixel is erased, check if darker or lighter than the CC average
                                    diff = cc_cut[y, x] - avg_cc_luminosity
                                    if diff > 0:
                                        reduction_pixels_lighter.append((diff, cc.min_y + y, cc.min_x + x))
                                    else:
                                        reduction_pixels_darker.append((-diff, cc.min_y + y, cc.min_x + x))

                        self.CC_shrinking_pixels_light = sorted(reduction_pixels_lighter, reverse=True)
                        self.CC_shrinking_pixels_dark = sorted(reduction_pixels_darker, reverse=True)

                        self.count_shrink_light = 0
                        self.count_shrink_dark = 0

                        self.expansion_scroll_lighter.reset(0, len(self.CC_shrinking_pixels_light), 0, 1)
                        self.expansion_scroll_darker.reset(0, len(self.CC_shrinking_pixels_dark), 0, 1)
                        self.set_editor_mode(GTPixelBinaryAnnotator.ModeShrinkCC_Shrinking)

                        # force one view
                        self.view_mode = GTPixelBinaryAnnotator.ViewModeSoftCombined
                        self.cc_show_mode = GTPixelBinaryAnnotator.CCShowBlack
                        self.update_current_view()
                        break

        elif button == 1 and self.editor_mode == GTPixelBinaryAnnotator.ModeDeleteCC_Select:
            for cc in self.base_ccs:
                if cc.min_x <= click_x <= cc.max_x and cc.min_y <= click_y <= cc.max_y:
                    rel_offset_x = click_x - cc.min_x
                    rel_offset_y = click_y - cc.min_y

                    if cc.img[rel_offset_y, rel_offset_x] > 0:
                        # cc is selected ... delete!
                        binary_cut = self.base_binary[cc.min_y:cc.max_y + 1, cc.min_x:cc.max_x + 1]
                        binary_cut[cc.img > 0] = 255

                        # finish edition
                        self.btn_edition_stop_click(self.btn_edition_stop)
                        self.update_current_view()
                        break

    def img_mouse_motion(self, img_object, pos, rel, buttons):
        if buttons[0] > 0 and self.editor_mode == GTPixelBinaryAnnotator.ModeEdition:

            scaled_x, scaled_y = pos
            move_x = int(scaled_x / self.view_scale)
            move_y = int(scaled_y / self.view_scale)

            if move_x < 0 or move_y < 0 or move_x >= self.base_raw.shape[1] or move_y >= self.base_raw.shape[0]:
                # out of boundaries
                return

            if (move_x != self.last_motion_set_x or move_y != self.last_motion_set_y) and (self.base_binary[move_y, move_x] != self.last_motion_polarity):
                # invert pixel ...
                self.base_binary[move_y, move_x] = 255 - self.base_binary[move_y, move_x]
                if self.base_binary[move_y, move_x] > 0:
                    self.base_cc_image[move_y, move_x, :] = 0
                else:
                    self.base_cc_image[move_y, move_x, 0] = 1
                    self.base_cc_image[move_y, move_x, 1] = 0
                    self.base_cc_image[move_y, move_x, 2] = 0

                self.last_motion_set_x = move_x
                self.last_motion_set_y = move_y

                self.update_current_view(False, (move_x - 1, move_x + 1, move_y - 1, move_y + 1))
        else:
            self.last_motion_set_x = None
            self.last_motion_set_y = None
            self.last_motion_polarity = None

    def btn_confirm_accept_click(self, button):
        if self.editor_mode == GTPixelBinaryAnnotator.ModeConfirmCancel:
            if self.finished_callback is not None:
                # accept ...
                self.finished_callback(False, None)

            self.return_screen = self.parent_screen

    def btn_confirm_cancel_click(self, button):
        # simply return to navigation mode mode
        self.set_editor_mode(GTPixelBinaryAnnotator.ModeNavigation)

    def btn_edition_expand_click(self, button):
        self.set_editor_mode(GTPixelBinaryAnnotator.ModeGrowCC_Select)

    def btn_edition_delete_click(self, button):
        # prepare for potential edit ...
        self.pre_edition_binary = self.base_binary.copy()
        self.set_editor_mode(GTPixelBinaryAnnotator.ModeDeleteCC_Select)

    def btn_edition_shrink_click(self, button):
        self.set_editor_mode(GTPixelBinaryAnnotator.ModeShrinkCC_Select)

    def btn_expand_accept_click(self, button):
        if self.editor_mode == GTPixelBinaryAnnotator.ModeGrowCC_Growing:
            # apply changes
            full_expansion = (self.CC_expansion_pixels_light[:self.count_expand_light] +
                              self.CC_expansion_pixels_dark[:self.count_expand_dark])
            for value, y, x in full_expansion:
                # mark as content ...
                self.base_binary[y, x] = 0

            self.CC_expansion_pixels_light = None
            self.CC_expansion_pixels_dark = None

        elif self.editor_mode == GTPixelBinaryAnnotator.ModeShrinkCC_Shrinking:
            # apply changes
            full_shrink = (self.CC_shrinking_pixels_light[:self.count_shrink_light] +
                           self.CC_shrinking_pixels_dark[:self.count_shrink_dark])
            for value, y, x in full_shrink:
                # mark as background ...
                self.base_binary[y, x] = 255

            self.CC_shrinking_pixels_light = None
            self.CC_shrinking_pixels_dark = None

        # clear selection
        self.selected_CC = None

        # finish edition
        self.btn_edition_stop_click(self.btn_edition_stop)
        self.update_current_view()

    def btn_expand_cancel_click(self, button):
        # clear selection
        self.selected_CC = None
        self.CC_expansion_pixels = None

        # go back to navigation mode
        self.set_editor_mode(GTPixelBinaryAnnotator.ModeNavigation)
        self.update_current_view()

    def update_label_expansion(self):
        if self.editor_mode == GTPixelBinaryAnnotator.ModeGrowCC_Growing:
            if len(self.CC_expansion_pixels_light) > 0:
                self.count_expand_light = self.expansion_scroll_lighter.value
                self.lbl_cc_expansion_lighter.set_text("Lighter Pixels = " + str(self.count_expand_light))
                self.expansion_scroll_lighter.visible = True
            else:
                # no lighter pixels where found in the CC expansion ....
                self.count_expand_light = 0
                self.expansion_scroll_lighter.visible = False

            self.lbl_cc_expansion_lighter.visible = self.expansion_scroll_lighter.visible

            if len(self.CC_expansion_pixels_dark) > 0:
                self.count_expand_dark = self.expansion_scroll_darker.value
                self.lbl_cc_expansion_darker.set_text("Darker Pixels = " + str(self.count_expand_dark))
                self.expansion_scroll_darker.visible = True
            else:
                # no darker pixels where found in the CC expansion ....
                self.count_expand_dark = 0
                self.expansion_scroll_darker.visible = False

            self.lbl_cc_expansion_darker.visible = self.expansion_scroll_darker.visible

        elif self.editor_mode == GTPixelBinaryAnnotator.ModeShrinkCC_Shrinking:
            if len(self.CC_shrinking_pixels_light) > 0:
                self.count_shrink_light = self.expansion_scroll_lighter.value
                self.lbl_cc_expansion_lighter.set_text("Lighter Pixels = " + str(self.count_shrink_light))
                self.expansion_scroll_lighter.visible = True
            else:
                self.count_shrink_light = 0
                self.expansion_scroll_lighter.visible = False

            self.lbl_cc_expansion_lighter.visible = self.expansion_scroll_lighter.visible

            if len(self.CC_shrinking_pixels_dark) > 0:
                self.count_shrink_dark = self.expansion_scroll_darker.value
                self.lbl_cc_expansion_darker.set_text("Darker Pixels = " + str(self.count_shrink_dark))
                self.expansion_scroll_darker.visible = True
            else:
                self.count_shrink_dark = 0
                self.expansion_scroll_darker.visible = False

            self.lbl_cc_expansion_darker.visible = self.expansion_scroll_darker.visible

    def expansion_scroll_change(self):
        self.update_label_expansion()

        start_x = self.selected_CC.min_x - GTPixelBinaryAnnotator.CCExpansionDistance
        end_x = self.selected_CC.max_x + GTPixelBinaryAnnotator.CCExpansionDistance
        start_y = self.selected_CC.min_y - GTPixelBinaryAnnotator.CCExpansionDistance
        end_y = self.selected_CC.max_y + GTPixelBinaryAnnotator.CCExpansionDistance

        self.update_current_view(False, (start_x, end_x, start_y, end_y))

    def expansion_scroll_lighter_change(self, scroll):
        self.expansion_scroll_change()

    def expansion_scroll_darker_change(self, scroll):
        self.expansion_scroll_change()


    def CC_apply_key_plus_or_minus(self, step):
        current_time = time.time()
        if (self.last_plus_minus_time is None or
            current_time - self.last_plus_minus_time >= GTPixelBinaryAnnotator.PlusMinusMultiTime):
            multiplier = 1
        else:
            multiplier = GTPixelBinaryAnnotator.PlusMinusMultiFactor

        """
        if self.last_plus_minus_time is not None:
            print(current_time - self.last_plus_minus_time)
        """

        self.last_plus_minus_time = current_time
        step *= multiplier

        if self.editor_mode == GTPixelBinaryAnnotator.ModeGrowCC_Growing:
            # Growing a CC ... by default try darker first ... then lighter ...
            if self.expansion_scroll_darker.visible:
                self.expansion_scroll_darker.apply_step(step)
                self.expansion_scroll_change()

            elif self.expansion_scroll_lighter.visible:
                self.expansion_scroll_lighter.apply_step(step)
                self.expansion_scroll_change()

        elif self.editor_mode == GTPixelBinaryAnnotator.ModeShrinkCC_Shrinking:
            # Shrinking a CC ... by default try lighter first ... then darker ...
            if self.expansion_scroll_lighter.visible:
                self.expansion_scroll_lighter.apply_step(step)
                self.expansion_scroll_change()
            elif self.expansion_scroll_darker.visible:
                self.expansion_scroll_darker.apply_step(step)
                self.expansion_scroll_change()

    def compute_dir_displacement_pixels(self, dil_kernel):
        # 1) compute pixels that will be added ...
        inv_binary = 255 - self.base_binary

        # ... 1.1) dilation on inverse binary (symbols are white)
        inv_dilation = cv2.dilate(inv_binary, dil_kernel)
        add_pixels = inv_dilation - inv_binary

        # ... 1.2) get corresponding pixels and their luminosities
        all_add_pos = np.nonzero(add_pixels)
        all_add_values = self.base_gray[all_add_pos][:, 0]

        # .... 1.3) pixels to add ... sorted by increasing luminosity (add darker first)
        pixels_to_add = [vals for vals in zip(all_add_values, all_add_pos[0], all_add_pos[1])]
        pixels_to_add = sorted(pixels_to_add)

        # 2) compute pixels that will be deleted ....
        # ... 2.1) dilation on raw binary (background is white)
        raw_dilation = cv2.dilate(self.base_binary, dil_kernel)
        del_pixels = raw_dilation - self.base_binary

        # ... 2.2) get corresponding pixels and their luminosities
        all_del_pos = np.nonzero(del_pixels)
        all_del_values = self.base_gray[all_del_pos][:, 0]

        # ... 2.3) pixels to remove ... sorted by decreasing luminosity (remove lighter first)
        pixels_to_remove = [vals for vals in zip(all_del_values, all_del_pos[0], all_del_pos[1])]
        pixels_to_remove = sorted(pixels_to_remove, reverse=True)

        # cv2.imshow("image", self.base_binary)
        # cv2.imshow("del", del_pixels)
        # cv2.imshow("add", add_pixels)
        # cv2.waitKey()

        return pixels_to_add, pixels_to_remove

    def compute_displacement_pixels(self, is_vertical):
        self.displacement_is_vertical = is_vertical
        self.displacement_percentage = 0
        self.displacement_pixels_adding = []
        self.displacement_pixels_removing = []
        self.pixelwise_displacement_scroll.reset(-100, 100, 0, 10)

        self.pre_edition_binary = self.base_binary.copy()

        if is_vertical:
            neg_dil_kernel = np.array([[0], [1], [1]], dtype=np.uint8)
            pos_dil_kernel = np.array([[1], [1], [0]], dtype=np.uint8)
        else:
            neg_dil_kernel = np.array([[0, 1, 1]], dtype=np.uint8)
            pos_dil_kernel = np.array([[1, 1, 0]], dtype=np.uint8)

        inv_binary = 255 - self.base_binary

        # affected pixels on the negative direction
        self.displacement_neg_add, self.displacement_neg_remove = self.compute_dir_displacement_pixels(neg_dil_kernel)

        # affected pixels on the positive direction
        self.displacement_pos_add, self.displacement_pos_remove = self.compute_dir_displacement_pixels(pos_dil_kernel)

    def btn_displacement_hor_click(self, button):
        self.compute_displacement_pixels(False)
        self.set_editor_mode(GTPixelBinaryAnnotator.ModePixelwise_Displacing)
        # force one view
        self.view_mode = GTPixelBinaryAnnotator.ViewModeSoftCombined
        self.cc_show_mode = GTPixelBinaryAnnotator.CCShowBlack
        self.update_current_view()

    def btn_displacement_ver_click(self, button):
        self.compute_displacement_pixels(True)
        self.set_editor_mode(GTPixelBinaryAnnotator.ModePixelwise_Displacing)

        # force one view
        self.view_mode = GTPixelBinaryAnnotator.ViewModeSoftCombined
        self.cc_show_mode = GTPixelBinaryAnnotator.CCShowBlack
        self.update_current_view()

    def pixelwise_displacement_scroll_change(self, scroll):
        # update ....
        self.displacement_percentage = scroll.value / 100.0

        if self.displacement_percentage < 0:
            # using the negative ....
            max_add = len(self.displacement_neg_add)
            max_del = len(self.displacement_neg_remove)
            n_to_add = int(max_add * self.displacement_percentage * -1.0)
            n_to_del = int(max_del * self.displacement_percentage * -1.0)

            self.displacement_pixels_adding = self.displacement_neg_add[:n_to_add]
            self.displacement_pixels_removing = self.displacement_neg_remove[:n_to_del]
        else:
            # using the positive ...
            max_add = len(self.displacement_pos_add)
            max_del = len(self.displacement_pos_remove)
            n_to_add = int(max_add * self.displacement_percentage)
            n_to_del = int(max_del * self.displacement_percentage)

            self.displacement_pixels_adding = self.displacement_pos_add[:n_to_add]
            self.displacement_pixels_removing = self.displacement_pos_remove[:n_to_del]

        add_msg = "Adding: {0:d} of {1:d}".format(len(self.displacement_pixels_adding), max_add)
        del_msg = "Removing: {0:d} of {1:d}".format(len(self.displacement_pixels_removing), max_del)

        self.lbl_pixelwise_disp_add.set_text(add_msg)
        self.lbl_pixelwise_disp_del.set_text(del_msg)

        self.update_current_view()


    def btn_pixelwise_displacement_accept_click(self, button):
        # apply changes
        # ... first remove pixels ....
        for value, y, x in self.displacement_pixels_removing:
            # mark as background ...
            self.base_binary[y, x] = 255

        # ... then add new pixels ....
        for value, y, x in self.displacement_pixels_adding:
            # mark as content ...
            self.base_binary[y, x] = 0

        # clear ...
        self.displacement_percentage = 0
        self.displacement_pixels_adding = None
        self.displacement_pixels_removing = None

        # finish edition
        self.set_editor_mode(GTPixelBinaryAnnotator.ModeNavigation)
        self.btn_edition_stop_click(self.btn_edition_stop)
        self.update_current_view()

    def btn_pixelwise_displacement_cancel_click(self, button):
        # go back to navigation mode
        self.set_editor_mode(GTPixelBinaryAnnotator.ModeNavigation)
        self.update_current_view()

    def main_key_up(self, scancode, key, unicode):
        # key short cuts
        if key == 269:
            # minus
            if self.container_view_buttons.visible:
                self.btn_zoom_reduce_click(None)
            elif self.editor_mode in [GTPixelBinaryAnnotator.ModeGrowCC_Growing,
                                      GTPixelBinaryAnnotator.ModeShrinkCC_Shrinking]:
                self.CC_apply_key_plus_or_minus(-1)

        elif key == 270:
            # plus
            if self.container_view_buttons.visible:
                self.btn_zoom_increase_click(None)
            elif self.editor_mode in [GTPixelBinaryAnnotator.ModeGrowCC_Growing,
                                      GTPixelBinaryAnnotator.ModeShrinkCC_Shrinking]:
                self.CC_apply_key_plus_or_minus(1)

        elif key == 13 or key == 271:
            # return key
            if self.container_confirm_buttons.visible:
                self.btn_confirm_accept_click(None)
            elif self.editor_mode == GTPixelBinaryAnnotator.ModeEdition:
                self.btn_edition_stop_click(None)
            elif self.container_cc_expansion.visible:
                self.btn_expand_accept_click(None)
            elif self.editor_mode == GTPixelBinaryAnnotator.ModeNavigation:
                    self.btn_return_accept_click(None)
        elif key == 27:
            # escape key ..
            if self.container_confirm_buttons.visible:
                self.btn_confirm_cancel_click(None)
            elif self.editor_mode == GTPixelBinaryAnnotator.ModeEdition:
                self.btn_edition_stop_click(None)
            elif self.container_cc_expansion.visible:
                self.btn_expand_cancel_click(None)
            elif self.editor_mode == GTPixelBinaryAnnotator.ModeNavigation:
                    self.btn_return_cancel_click(None)

        elif key == 100:
            # D - Delete CC
            if self.editor_mode == GTPixelBinaryAnnotator.ModeNavigation:
                self.btn_edition_delete_click(None)

        elif key == 101:
            # E - Expand CC
            if self.editor_mode == GTPixelBinaryAnnotator.ModeNavigation:
                self.btn_edition_expand_click(None)

        elif key == 112:
            # P - Edit Pixels
            if self.editor_mode == GTPixelBinaryAnnotator.ModeNavigation:
                self.btn_edition_start_click(None)

        elif key == 115:
            # S - Shrink CC
            if self.editor_mode == GTPixelBinaryAnnotator.ModeNavigation:
                self.btn_edition_shrink_click(None)

        else:
            # print(key)
            pass

    def img_double_click(self, control, pos, button):
        if button == 1:
            if self.editor_mode == GTPixelBinaryAnnotator.ModeNavigation:
                self.btn_edition_start_click(None)
        elif button == 3:
            if self.editor_mode == GTPixelBinaryAnnotator.ModeEdition:
                self.btn_edition_stop_click(None)
