
import random

import cv2
import numpy as np
from PIL import Image

import torch
import torch.utils
import torch.utils.data

from torchvision import transforms
import torchvision.transforms.functional as TF


class LectureNet_DataSet(torch.utils.data.Dataset):
    def __init__(self, image_list, ground_truth_list, reconstruction_mode,
                 crop_size=None, crop_remove_empty_borders=False, crop_min_fg_prc=None, flip_chance=None,
                 color_invert_chance=None, color_change_chance=None, luminosity_changes_chance=None,
                 gaussian_noise_chance=None, gaussian_noise_range=None,
                 weight_expansion=None, weight_fg_extra=None,
                 text_region_masks_expansion=None,
                 reconstruct_median=False, reconstruct_median_K=None):

        if ground_truth_list is not None:
            assert len(image_list) == len(ground_truth_list)

        self.image_list = image_list
        self.ground_truth_list = ground_truth_list

        self.reconstruction_mode = reconstruction_mode

        self.crop_size = crop_size
        self.crop_remove_empty_borders = crop_remove_empty_borders
        self.crop_min_fg_prc = crop_min_fg_prc

        self.flip_chance = flip_chance

        self.color_invert_chance = color_invert_chance

        self.color_change_chance = color_change_chance
        self.luminosity_changes_chance = luminosity_changes_chance

        self.gaussian_noise_chance = gaussian_noise_chance
        self.gaussian_noise_range = gaussian_noise_range

        self.weight_expansion = weight_expansion
        self.weight_fg_extra = weight_fg_extra
        if weight_expansion is None:
            self.weight_st_element = None
        else:
            disk_size = (self.weight_expansion * 2 + 1, self.weight_expansion * 2 + 1)
            self.weight_st_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, disk_size )

        self.text_region_masks_expansion = text_region_masks_expansion
        if text_region_masks_expansion is None:
            self.text_region_mask_st = None
        else:
            disk_size = (self.text_region_masks_expansion * 2 + 1, self.text_region_masks_expansion * 2 + 1)
            self.text_region_mask_st = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, disk_size)

        self.total_foreground = None
        self.total_background = None

        self.preloaded_images = None
        self.preloaded_ground_truths = None

        self.reconstruct_median = reconstruct_median
        self.reconstruct_median_K = reconstruct_median_K

    def load_image_pair(self, img_filename, gt_filename):
        img = cv2.imread(img_filename)
        # for later usage with PIL
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if gt_filename is not None:
            gt_img = cv2.imread(gt_filename)
            # for later usage as GT
            gt_img = gt_img[:, :, 0]
        else:
            gt_img = None

        h, w, _ = img.shape

        if self.crop_remove_empty_borders and gt_img is not None:
            # find region that has text
            horizontal_range = np.nonzero((255 - gt_img).max(axis=0))[0]
            x_min_val = horizontal_range[0]
            x_max_val = horizontal_range[-1]

            vertical_range = np.nonzero((255 - gt_img).max(axis=1))[0]
            y_min_val = vertical_range[0]
            y_max_val = vertical_range[-1]

            half_w_margin = 10
            half_h_margin = 10

            start_x = max(0, x_min_val - half_w_margin)
            end_x = min(w, x_max_val + half_w_margin)

            start_y = max(0, y_min_val - half_h_margin)
            end_y = min(h, y_max_val + half_h_margin)

            # check ...
            if self.crop_size is not None:
                if end_x - start_x < self.crop_size[1]:
                    mid_point = int((start_x + end_x) / 2)
                    start_x = max(0, mid_point - int(self.crop_size[1] / 2 + 1))
                    end_x = min(w, start_x + self.crop_size[1])

                if end_y - start_y < self.crop_size[0]:
                    mid_point = int((start_y + end_y) / 2)
                    start_y = max(0, mid_point - int(self.crop_size[0] / 2 + 1))
                    end_y = min(h, start_y + self.crop_size[0])

            img = img[start_y:end_y, start_x:end_x]
            gt_img = gt_img[start_y:end_y, start_x:end_x]

            # update
            h, w, _ = img.shape

        # check for auto-resize (if the image is too small)
        if self.crop_size is not None and (h < self.crop_size[0] or w < self.crop_size[1]):
            # resize the image
            w_scale_factor = self.crop_size[1] / w
            h_scale_factor = self.crop_size[0] / h

            if w_scale_factor > h_scale_factor:
                # upscale by width
                new_height = int(round(h * w_scale_factor))
                new_width = self.crop_size[1]
            else:
                new_height = self.crop_size[0]
                new_width = int(round(w * h_scale_factor))

            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            if gt_img is not None:
                gt_img = cv2.resize(gt_img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        # cv2.imwrite("ZZZ.png", img)
        # cv2.waitKey()
        # x = 0 / 0
        return img, gt_img

    def preload(self, verbose=True):
        self.total_foreground = 0.0
        self.total_background = 0.0
        self.preloaded_images = []
        if self.ground_truth_list is not None:
            self.preloaded_ground_truths = []

        for img_idx, img_filename in enumerate(self.image_list):
            if verbose:
                print(" " * 120, end="\r")
                print("{0:d} - {1:s}".format(img_idx + 1, img_filename), end="\r")

            if self.ground_truth_list is not None:
                gt_filename = self.ground_truth_list[img_idx]
            else:
                gt_filename = None

            img, gt_img = self.load_image_pair(img_filename, gt_filename)

            # count background and foreground pixels
            image_foreground = (gt_img == 0).sum()
            self.total_foreground += image_foreground
            self.total_background += (gt_img.size - image_foreground)

            # encode and store
            # ... main image ...
            flag, encoded_img = cv2.imencode(".png", img)
            self.preloaded_images.append(encoded_img)
            # ... gt image ....
            if self.ground_truth_list is not None:
                flag, encoded_img = cv2.imencode(".png", gt_img)
                self.preloaded_ground_truths.append(encoded_img)

        if verbose:
            print("Image pre-loading complete!")

    def __len__(self):
        return len(self.image_list)

    def get_full_image(self, index, with_gt=False):
        if self.preloaded_images is not None:
            # a compressed copy is stored on RAM ... decode and use!
            compresed_img = self.preloaded_images[index]
            raw_img = cv2.imdecode(compresed_img, cv2.IMREAD_COLOR)

            pil_img = Image.fromarray(raw_img)

            if not with_gt:
                return pil_img
            else:
                compressed_gt = self.preloaded_ground_truths[index]
                raw_gt = cv2.imdecode(compressed_gt, cv2.IMREAD_GRAYSCALE)

                pil_gt = Image.fromarray(raw_gt)

                return pil_img, pil_gt
        else:
            # load from Disk (slower!)
            img_filename = self.image_list[index]
            if self.ground_truth_list is not None:
                gt_filename = self.ground_truth_list[index]
            else:
                gt_filename = None

            raw_img, raw_gt = self.load_image_pair(img_filename, gt_filename)

            pil_img = Image.fromarray(raw_img)

            if not with_gt:
                return pil_img
            else:
                pil_gt = Image.fromarray(raw_gt)

                return pil_img, pil_gt

    def __getitem__(self, index):
        if self.ground_truth_list is not None:
            pil_img, pil_gt = self.get_full_image(index, True)
        else:
            pil_img = self.get_full_image(index, False)
            pil_gt = None

        if self.flip_chance is not None:
            # try horizontal flipping
            if random.random() < self.flip_chance:
                pil_img = TF.hflip(pil_img)
                if pil_gt is not None:
                    pil_gt = TF.hflip(pil_gt)

            # try vertical flipping
            if random.random() < self.flip_chance:
                pil_img = TF.vflip(pil_img)
                if pil_gt is not None:
                    pil_gt = TF.vflip(pil_gt)

        # do cropping
        if self.crop_size is not None:
            valid_crop = False
            n_crop_tests = 0
            while not valid_crop:
                i, j, h, w = transforms.RandomCrop.get_params(pil_img, output_size=self.crop_size)
                tempo_crop_img = TF.crop(pil_img, i, j, h, w)
                if pil_gt is not None:
                    tempo_crop_gt = TF.crop(pil_gt, i, j, h, w)
                else:
                    # no ground truth for validation ... assume valid
                    tempo_crop_gt = None
                    valid_crop = True

                if self.crop_min_fg_prc is not None:
                    crop_fg_percentage = (np.asarray(tempo_crop_gt) == 0).sum() / (self.crop_size[0] * self.crop_size[1])
                    # print(crop_fg_percentage)
                    valid_crop = crop_fg_percentage >= self.crop_min_fg_prc
                else:
                    # no validation, just stop here
                    valid_crop = True

                if n_crop_tests > 5:
                    # it has been tested more than 10 times without success...
                    #  keep it because the frame might not contain any valid crop
                    valid_crop = True

                if valid_crop:
                    # it is either valid or
                    pil_img = tempo_crop_img
                    pil_gt = tempo_crop_gt
                else:
                    n_crop_tests += 1

        if self.color_invert_chance is not None and random.random() < self.color_invert_chance:
            # invert colors
            img_np = np.asarray(pil_img)
            img_np = 255 - img_np
            pil_img = Image.fromarray(img_np)

        if self.color_change_chance is not None and random.random() < self.color_change_chance:

            # transform color by using HUE
            pil_img = TF.adjust_hue(pil_img, (random.random() * 0.9 - 0.45))

        if self.gaussian_noise_chance is not None and random.random() < self.gaussian_noise_chance:
            # add gaussian noise
            img_np = np.asarray(pil_img).astype(np.float64)
            img_np += np.random.randn(img_np.shape[0], img_np.shape[1], img_np.shape[2]) * self.gaussian_noise_range
            img_np[img_np < 0] = 0
            img_np[img_np > 255] = 255
            pil_img = Image.fromarray(img_np.astype(np.uint8))

        if self.luminosity_changes_chance is not None and random.random() < self.luminosity_changes_chance:
            # Apply random changes that affect the luminosity and Sharpness of the image

            if np.random.randn() < 0:
                # lower brightness ... uniform ... from 0.75 to 1.0
                pil_img = TF.adjust_brightness(pil_img, 1.0 - np.random.rand() * 0.25)
            else:
                # increase brightness ... uniform ... from 1.0 to 1.5
                pil_img = TF.adjust_brightness(pil_img, 1.0 + np.random.rand() * 0.50)

            if np.random.randn() < 0:
                # lower contrast ... uniform ... from 0.50 to 1.0
                pil_img = TF.adjust_contrast(pil_img, 1.0 - np.random.rand() * 0.5)
            else:
                # increase contrast ... uniform ... from 1.0 to 2.0
                pil_img = TF.adjust_contrast(pil_img, 1.0 + np.random.rand() * 1.0)

            if np.random.randn() < 0:
                # lower gamma ... uniform ... from 0.50 to 1.0
                pil_img = TF.adjust_gamma(pil_img, 1.0 - np.random.rand() * 0.50)
            else:
                # increase gamma ... uniform ... from 1.0 to 2.0
                pil_img = TF.adjust_gamma(pil_img, 1.0 + np.random.rand() * 1.00)

            if np.random.randn() < 0:
                # lower the saturation ... uniform ... between 0.25 to 1.0 saturation
                pil_img = TF.adjust_saturation(pil_img, 1.0 - np.random.rand() * 0.75)
            else:
                # increase the saturation ... uniform ... between 1.0 to 5.0
                pil_img = TF.adjust_saturation(pil_img, 1.0 + np.random.rand() * 4.0)

        if self.text_region_mask_st is not None:
            mask_gt = np.asarray(pil_gt)
            mask_gt = 255 - cv2.erode(mask_gt, self.text_region_mask_st)
            mask_gt = Image.fromarray(mask_gt)

            text_mask_gt_t = TF.to_tensor(mask_gt)
        else:
            text_mask_gt_t = 0

        """
        # debugging
        debug_img = np.asarray(pil_img)
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
        debug_gt = np.asarray(pil_gt)
        cv2.imshow("Image", debug_img)
        cv2.imshow("GT", debug_gt)
        cv2.imshow("mask", mask_gt)
        cv2.waitKey()
        """

        # convert to tensor
        img_t = TF.to_tensor(pil_img)

        if self.reconstruct_median:
            median_img = np.asarray(pil_img)
            median_img = cv2.medianBlur(median_img, self.reconstruct_median_K)
            median_pil = Image.fromarray(median_img)
            median_t = TF.to_tensor(median_pil)

            # color normalization
            # median_t = TF.normalize(median_t, [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # simple output rage adjustment
            median_t = TF.normalize(median_t, [0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            median_t = 0

        if self.reconstruction_mode:
            gt_t = img_t

            # same normalization as input
            # ... RGB normalization ....
            # gt_t = TF.normalize(gt_t, [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # ... color adjustment ....
            gt_t = TF.normalize(img_t, [0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            # convert the mask to tensor
            if pil_gt is not None:
                gt_t = TF.to_tensor(pil_gt)
            else:
                gt_t = 0

        # ... normalize the RGB values
        # img_t = TF.normalize(img_t, [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ... no normalization ... simply re-scale to Tanh range ...
        img_t = TF.normalize(img_t, [0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        # check if per pixel weights are required ....
        if self.weight_st_element is not None:
            # get weights ....
            weight_gt = np.asarray(pil_gt)
            eroded_gt = cv2.erode(weight_gt, self.weight_st_element)

            # start with uniform weights
            weights = np.ones(eroded_gt.shape, dtype=np.float64)
            # add extra proportion
            weights[eroded_gt == 0] += self.weight_fg_extra

            """
            print(total_fg_exp_pixels)
            print(fg_proportion)
            print(weights.min())
            print(weights.max())
            vis_weights = ((weights / weights.max()) * 255).astype(dtype=np.uint8)

            cv2.imshow("original", weight_gt)
            cv2.imshow("eroded", eroded_gt)
            cv2.imshow("weights", vis_weights)
            cv2.waitKey()
            """

            weights_t = torch.tensor(weights)
        else:
            # no weights
            weights_t = 0

        return img_t, gt_t, weights_t, text_mask_gt_t, median_t
