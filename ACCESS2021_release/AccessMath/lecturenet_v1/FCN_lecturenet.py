
import cv2
import numpy as np

import PIL
import PIL.Image

import torch
import torch.nn as nn

from torchvision import transforms
import torchvision.transforms.functional as TF

from AM_CommonTools.configuration.configuration import Configuration

class FCN_LectureNet(nn.Module):
    def __init__(self, channels, n_conv_down_1, n_conv_down_2, n_conv_down_3, n_conv_down_4, n_conv_down_5, mid_block,
                 n_upsample_5, n_conv_up_5, n_upsample_4, n_conv_up_4, n_upsample_3, n_conv_up_3,
                 n_upsample_2, n_conv_up_2, n_upsample_1, n_conv_up_1, kernel_size,
                 n_pmaps_1, n_pmaps_2, pixel_kernel_size, reconstruction_mode):
        super(FCN_LectureNet, self).__init__()

        # initial convolutions ...
        padding = int((kernel_size - 1) / 2)

        self.conv_down_block_1 = nn.Sequential(
            nn.Conv2d(channels, n_conv_down_1, stride=1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(n_conv_down_1),
            nn.GELU()
        )
        self.conv_block_pool_1 = nn.MaxPool2d(2, return_indices=False)
        nn.init.xavier_normal_(self.conv_down_block_1[0].weight)

        self.conv_down_block_2 = nn.Sequential(
            nn.Conv2d(n_conv_down_1, n_conv_down_2, stride=1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(n_conv_down_2),
            nn.GELU(),
        )
        self.conv_block_pool_2 = nn.MaxPool2d(2, return_indices=False)
        nn.init.xavier_normal_(self.conv_down_block_2[0].weight)

        self.conv_down_block_3 = nn.Sequential(
            nn.Conv2d(n_conv_down_2, n_conv_down_3, stride=1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(n_conv_down_3),
            nn.GELU()
        )
        self.conv_block_pool_3 = nn.MaxPool2d(2, return_indices=False)
        nn.init.xavier_normal_(self.conv_down_block_3[0].weight)

        self.conv_down_block_4 = nn.Sequential(
            nn.Conv2d(n_conv_down_3, n_conv_down_4, stride=1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(n_conv_down_4),
            nn.GELU()
        )
        self.conv_block_pool_4 = nn.MaxPool2d(2, return_indices=False)
        nn.init.xavier_normal_(self.conv_down_block_4[0].weight)

        self.conv_down_block_5 = nn.Sequential(
            nn.Conv2d(n_conv_down_4, n_conv_down_5, stride=1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(n_conv_down_5),
            nn.GELU()
        )
        self.conv_block_pool_5 = nn.MaxPool2d(2, return_indices=False)
        nn.init.xavier_normal_(self.conv_down_block_5[0].weight)


        # middle conv block
        self.mid_block = nn.Sequential(
            nn.Conv2d(n_conv_down_5, mid_block, stride=1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(mid_block),
            nn.GELU(),
        )
        nn.init.xavier_normal_(self.mid_block[0].weight)


        self.transposed_conv_5 = nn.ConvTranspose2d(mid_block, n_upsample_5, 2, padding=0, stride=2)
        self.upsample_block_5 = nn.Sequential(
            nn.BatchNorm2d(n_upsample_5),
            nn.GELU()
        )
        nn.init.xavier_normal_(self.transposed_conv_5.weight)
        self.conv_up_block_5 = nn.Sequential(
            nn.Conv2d(n_upsample_5 + n_conv_down_5, n_conv_up_5, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(n_conv_up_5),
            nn.GELU()
        )
        nn.init.xavier_normal_(self.conv_up_block_5[0].weight)

        self.transposed_conv_4 = nn.ConvTranspose2d(n_conv_up_5, n_upsample_4, 2, padding=0, stride=2)
        self.upsample_block_4 = nn.Sequential(
            nn.BatchNorm2d(n_upsample_4),
            nn.GELU()
        )
        nn.init.xavier_normal_(self.transposed_conv_4.weight)
        self.conv_up_block_4 = nn.Sequential(
            nn.Conv2d(n_upsample_4 + n_conv_down_4, n_conv_up_4, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(n_conv_up_4),
            nn.GELU()
        )
        nn.init.xavier_normal_(self.conv_up_block_4[0].weight)

        self.transposed_conv_3 = nn.ConvTranspose2d(n_conv_up_4, n_upsample_3, 2, padding=0, stride=2)
        self.upsample_block_3 = nn.Sequential(
            nn.BatchNorm2d(n_upsample_3),
            nn.GELU()
        )
        nn.init.xavier_normal_(self.transposed_conv_3.weight)
        self.conv_up_block_3 = nn.Sequential(
            nn.Conv2d(n_upsample_3 + n_conv_down_3, n_conv_up_3, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(n_conv_up_3),
            nn.GELU()
        )
        nn.init.xavier_normal_(self.conv_up_block_3[0].weight)

        self.transposed_conv_2 = nn.ConvTranspose2d(n_conv_up_3, n_upsample_2, 2, padding=0, stride=2)
        self.upsample_block_2 = nn.Sequential(
            nn.BatchNorm2d(n_upsample_2),
            nn.GELU()
        )
        nn.init.xavier_normal_(self.transposed_conv_2.weight)
        self.conv_up_block_2 = nn.Sequential(
            nn.Conv2d(n_upsample_2 + n_conv_down_2, n_conv_up_2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(n_conv_up_2),
            nn.GELU()
        )
        nn.init.xavier_normal_(self.conv_up_block_2[0].weight)

        self.transposed_conv_1 = nn.ConvTranspose2d(n_conv_up_2, n_upsample_1, 2, padding=0, stride=2)
        self.upsample_block_1 = nn.Sequential(
            nn.BatchNorm2d(n_upsample_1),
            nn.GELU()
        )
        nn.init.xavier_normal_(self.transposed_conv_1.weight)
        self.conv_up_block_1 = nn.Sequential(
            nn.Conv2d(n_upsample_1 + n_conv_down_1, n_conv_up_1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(n_conv_up_1),
            nn.GELU()
        )
        nn.init.xavier_normal_(self.conv_up_block_1[0].weight)

        # FIRST OUTPUT BRANCH: BINARIZATION
        self.conv_pixels_1 = None
        self.conv_pixels_2 = None
        # output 1: binary
        self.conv_out = None

        # SECOND OUTPUT BRANCH: Text Masks
        self.conv_text_mask_out = None

        self.set_main_branches(channels, pixel_kernel_size, n_conv_up_1, n_pmaps_1, n_pmaps_2)

        # THIRD OUTPUT BRANCH: reconstruction
        self.conv_reconstruct = nn.Sequential(
            nn.Conv2d(n_conv_up_1, 3, stride=1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(3),
            # nn.Sigmoid()
            # nn.GELU(),
            nn.Tanh(),
        )
        nn.init.xavier_normal_(self.conv_reconstruct[0].weight)

        self.reconstruction_mode = reconstruction_mode

    def set_main_branches(self, channels, kernel_size,  n_conv_up_1, n_pmaps_1, n_pmaps_2):
        padding = int((kernel_size - 1) / 2)

        # FIRST OUTPUT BRANCH: BINARIZATION
        # .... now use 1D convolutions ...
        inputs_conv_pixels_1 = channels + n_conv_up_1
        # inputs_conv_pixels_1 = channels + n_conv_up_1 + 1
        self.conv_pixels_1 = nn.Sequential(
            nn.Conv2d(inputs_conv_pixels_1, n_pmaps_1, stride=1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(n_pmaps_1),
            nn.GELU()
        )
        nn.init.xavier_normal_(self.conv_pixels_1[0].weight)

        inputs_conv_pixels_2 = channels + n_pmaps_1
        # inputs_conv_pixels_2 = channels + n_pmaps_1 + 1
        self.conv_pixels_2 = nn.Sequential(
            nn.Conv2d(inputs_conv_pixels_2, n_pmaps_2, stride=1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(n_pmaps_2),
            nn.GELU()
        )
        nn.init.xavier_normal_(self.conv_pixels_2[0].weight)

        # output 1: binary
        inputs_conv_pixels_3 = channels + n_pmaps_2
        # inputs_conv_pixels_3 = channels + n_pmaps_2 + 1
        self.conv_out = nn.Sequential(
            nn.Conv2d(inputs_conv_pixels_3, 1, stride=1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(1)
        )
        nn.init.xavier_normal_(self.conv_out[0].weight)

        # SECOND OUTPUT BRANCH: Text Masks
        self.conv_text_mask_out = nn.Sequential(
            nn.Conv2d(n_conv_up_1, 1, stride=1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(1)
        )
        nn.init.xavier_normal_(self.conv_text_mask_out[0].weight)

    def reset_main_branches(self, in_channels, config):
        n_convs_up_1 = config.get("FCN_BINARIZER_NET_UP_CONV_FILTERS_1", 16)

        n_pmaps_1 = config.get("FCN_BINARIZER_NET_PIXEL_FEATURES_1", 32)
        n_pmaps_2 = config.get("FCN_BINARIZER_NET_PIXEL_FEATURES_2", 16)

        pix_kernel_size = config.get("FCN_BINARIZER_NET_PIXEL_KERNEL_SIZE", 3)

        self.set_main_branches(in_channels, pix_kernel_size, n_convs_up_1, n_pmaps_1, n_pmaps_2)

    def get_batch_mid_block_features(self, batch_img):
        using_cuda = next(self.parameters()).is_cuda

        if using_cuda:
            batch_img = batch_img.cuda(0)

        with torch.no_grad():
            x_conv1 = self.conv_block_pool_1(self.conv_down_block_1(batch_img))
            x_conv2 = self.conv_block_pool_2(self.conv_down_block_2(x_conv1))
            x_conv3 = self.conv_block_pool_3(self.conv_down_block_3(x_conv2))
            x_conv4 = self.conv_block_pool_4(self.conv_down_block_4(x_conv3))
            x_conv5 = self.conv_block_pool_5(self.conv_down_block_5(x_conv4))

            x_mid = self.mid_block(x_conv5)

        if using_cuda:
            x_mid = x_mid.cpu()

        x_mid = x_mid.numpy()

        return x_mid

    def get_mid_block_features(self, PIL_image):
        # prepare ...
        batch_img = FCN_LectureNet.prepare_image(PIL_image)

        using_cuda = next(self.parameters()).is_cuda

        if using_cuda:
            batch_img = batch_img.cuda(0)

        with torch.no_grad():
            x_conv1 = self.conv_block_pool_1(self.conv_down_block_1(batch_img))
            x_conv2 = self.conv_block_pool_2(self.conv_down_block_2(x_conv1))
            x_conv3 = self.conv_block_pool_3(self.conv_down_block_3(x_conv2))
            x_conv4 = self.conv_block_pool_4(self.conv_down_block_4(x_conv3))
            x_conv5 = self.conv_block_pool_5(self.conv_down_block_5(x_conv4))

            x_mid = self.mid_block(x_conv5)

        if using_cuda:
            x_mid = x_mid.cpu()

        x_mid = x_mid[0].numpy()

        return x_mid

    def encode_decode(self, x0):
        # ... first architecture ... NO downsizing
        # x_conv1, indices_p1 = self.conv_block_1(x0)
        x_conv1_pre = self.conv_down_block_1(x0)
        x_conv1 = self.conv_block_pool_1(x_conv1_pre)

        x_conv2_pre = self.conv_down_block_2(x_conv1)
        x_conv2 = self.conv_block_pool_2(x_conv2_pre)

        x_conv3_pre = self.conv_down_block_3(x_conv2)
        x_conv3 = self.conv_block_pool_3(x_conv3_pre)

        x_conv4_pre = self.conv_down_block_4(x_conv3)
        x_conv4 = self.conv_block_pool_4(x_conv4_pre)

        x_conv5_pre = self.conv_down_block_5(x_conv4)
        x_conv5 = self.conv_block_pool_5(x_conv5_pre)

        x_mid = self.mid_block(x_conv5)

        x_up5 = self.transposed_conv_5(x_mid, output_size=x_conv4.shape)
        x_up5 = self.upsample_block_5(x_up5)
        x_up5 = torch.cat((x_up5, x_conv5_pre), 1)
        x_up5 = self.conv_up_block_5(x_up5)

        x_up4 = self.transposed_conv_4(x_up5, output_size=x_conv3.shape)
        x_up4 = self.upsample_block_4(x_up4)
        x_up4 = torch.cat((x_up4, x_conv4_pre), 1)
        x_up4 = self.conv_up_block_4(x_up4)

        x_up3 = self.transposed_conv_3(x_up4, output_size=x_conv2.shape)
        x_up3 = self.upsample_block_3(x_up3)
        x_up3 = torch.cat((x_up3, x_conv3_pre), 1)
        x_up3 = self.conv_up_block_3(x_up3)

        x_up2 = self.transposed_conv_2(x_up3, output_size=x_conv1.shape)
        x_up2 = self.upsample_block_2(x_up2)
        x_up2 = torch.cat((x_up2, x_conv2_pre), 1)
        x_up2 = self.conv_up_block_2(x_up2)

        x_up1 = self.transposed_conv_1(x_up2, output_size=x0.shape)
        x_up1 = self.upsample_block_1(x_up1)
        x_up1 = torch.cat((x_up1, x_conv1_pre), 1)
        x_up1 = self.conv_up_block_1(x_up1)

        """
        print("Going down ...")
        print(x0.shape)
        print(x_conv1.shape)
        print(x_conv2.shape)
        print(x_conv3.shape)
        print(x_conv4.shape)

        print("... middle ...")
        print(x_mid.shape)

        print("Going up ...")
        print(x_up4.shape)
        print(x_up3.shape)
        print(x_up2.shape)
        print(x_up1.shape)
        """

        return x_up1

    def get_batch_diff_images(self, batch_img, concat_features, downsample=None):
        using_cuda = next(self.parameters()).is_cuda

        if using_cuda:
            batch_img = batch_img.cuda(0)

        with torch.no_grad():
            x_up1 = self.encode_decode(batch_img)

            text_mask = self.conv_text_mask_out(x_up1)
            bin_text_mask = torch.sigmoid(text_mask)

            # using reconstruction branch ...
            rec_img = self.conv_reconstruct(x_up1)
            diff_img = (batch_img - rec_img) * bin_text_mask

            if concat_features:
                diff_img = torch.cat((diff_img, x_up1), 1)

            if downsample is not None:
                diff_img = nn.functional.max_pool2d(diff_img, kernel_size=downsample)

        if using_cuda:
            diff_img = diff_img.cpu()

        diff_img = diff_img.numpy()

        return diff_img

    def get_diff_image(self, PIL_image, concat_features, downsample=None):
        # prepare image and make it into batch
        batch_img = FCN_LectureNet.prepare_image(PIL_image)

        diff_img = self.get_batch_diff_images(batch_img, concat_features, downsample)

        diff_img = diff_img[0]

        return diff_img

    def forward(self, x0):
        x_up1 = self.encode_decode(x0)

        if not self.reconstruction_mode:
            # using text mask branch
            # Second Branch: Text Mask
            text_mask = self.conv_text_mask_out(x_up1)

            bin_text_mask = torch.sigmoid(text_mask)

            # using reconstruction branch ...
            # rec_img = self.conv_reconstruct(x_up1) * 2.75
            rec_img = self.conv_reconstruct(x_up1)
            diff_img = (x0 - rec_img) * bin_text_mask

            # First Branch: Binary Image
            # ... add the input maps ...
            # x_pixels_0 = torch.cat((x0, x_up1), 1)
            # x_pixels_0 = torch.cat((x0, bin_text_mask, x_up1), 1)
            x_pixels_0 = torch.cat((diff_img, x_up1), 1)

            # ... apply 1x1 convolution #1
            x_pixels_1 = self.conv_pixels_1(x_pixels_0)
            # ... add the input maps ...
            # x_pixels_1 = torch.cat((x0, x_pixels_1), 1)
            # x_pixels_1 = torch.cat((x0, bin_text_mask, x_pixels_1), 1)
            x_pixels_1 = torch.cat((diff_img, x_pixels_1), 1)

            # ... apply 1x1 convolution #2
            x_pixels_2 = self.conv_pixels_2(x_pixels_1)
            # ... add the input maps ...
            # x_pixels_2 = torch.cat((x0, x_pixels_2), 1)
            # x_pixels_2 = torch.cat((x0, bin_text_mask, x_pixels_2), 1)
            x_pixels_2 = torch.cat((diff_img, x_pixels_2), 1)

            # ... get last combination (NO SIGMOID)
            output = self.conv_out(x_pixels_2)

            # return output, text_mask
            return output, text_mask, rec_img
        else:
            # using reconstruction mode ... output from reconstruction branch
            raw_img = self.conv_reconstruct(x_up1)

            # this output was generated by a sigmoid ... it comes in range 0 to 1 same as raw PIL image
            # need to apply the same normalization to make it directly comparable to input

            """
            Sigmoid ... normalize adjust output... 
            red_p = torch.unsqueeze((raw_img[:, 0] - 0.485) / 0.229, 1)
            green_p = torch.unsqueeze((raw_img[:, 1] - 0.456) / 0.224, 1)
            blue_p = torch.unsqueeze((raw_img[:, 2] - 0.406) / 0.225, 1)
            
            output = torch.cat((red_p, green_p, blue_p), 1)
            """

            # For Tanh ... simply produce output in range -1 to 1
            output = raw_img

            # Tanh is from -1 to 1 .... stretch to produce range of inputs
            # ... this will allow values between (-2.75 to 2.75)
            # output = raw_img * 2.75

            return output


    def binarize(self, PIL_image, return_others=False, force_binary=False, binary_treshold=128, apply_sigmoid=True):
        o_width, o_height = PIL_image.size
        width = o_width
        height = o_height
        # check for images bigger than 2.5 Mega Pixels
        while width * height > 2500000:
            PIL_image = PIL_image.resize((int(width / 2), int(height / 2)), PIL.Image.LANCZOS)
            width, height = PIL_image.size

        # assert isinstance(PIL_image, Image)
        batch_img = FCN_LectureNet.prepare_image(PIL_image)

        using_cuda = next(self.parameters()).is_cuda

        if using_cuda:
            batch_img = batch_img.cuda(0)

        with torch.no_grad():
            # res, text_mask = self.forward(batch_img)
            res, text_mask, rec_img = self.forward(batch_img)

            # applying sigmoid
            if apply_sigmoid:
                res = torch.sigmoid(res)
                text_mask = torch.sigmoid(text_mask)

        if using_cuda:
            res = res.cpu()
            text_mask = text_mask.cpu()
            rec_img = rec_img.cpu()

        binary = res[0, 0].numpy() * 255
        binary = binary.astype(np.uint8)

        if force_binary:
            # use hard threshold
            binary[binary >= binary_treshold] = 255
            binary[binary < binary_treshold] = 0

        if return_others:
            text_mask = text_mask[0, 0].numpy() * 255
            text_mask = text_mask.astype(np.uint8)

            if force_binary:
                # use hard threshold
                text_mask[text_mask >= binary_treshold] = 255
                text_mask[text_mask < binary_treshold] = 0

            rec_img = rec_img[0].numpy()
            rec_img = self.from_img_space_to_cv2(rec_img)

        if o_width != width:
            # need to resize them again ....
            if force_binary:
                binary = cv2.resize(binary, (o_width, o_height), interpolation=cv2.INTER_NEAREST)
            else:
                binary = cv2.resize(binary, (o_width, o_height), interpolation=cv2.INTER_CUBIC)

            if return_others:
                if force_binary:
                    text_mask = cv2.resize(text_mask, (o_width, o_height), interpolation=cv2.INTER_NEAREST)
                else:
                    text_mask = cv2.resize(text_mask, (o_width, o_height), interpolation=cv2.INTER_CUBIC)

                rec_img = cv2.resize(rec_img, (o_width, o_height), interpolation=cv2.INTER_NEAREST)

        """
        cv2.imshow("check", binary)
        if return_mask:
            cv2.imshow("mask", text_mask)
        cv2.waitKey()
        """
        if return_others:
            return binary, text_mask, rec_img
        else:
            return binary

    def from_img_space_to_cv2_sigmoid(self, image):
        # assume input is normalized numpy array in the same format as it is used for input images

        # from PIL to CV2 ... transpose
        image = np.transpose(image, (1, 2, 0))

        image[:, :, 0] *= 0.229
        image[:, :, 1] *= 0.224
        image[:, :, 2] *= 0.225

        image[:, :, 0] += 0.485
        image[:, :, 1] += 0.456
        image[:, :, 2] += 0.406

        # swap channels
        tempo = image[:, :, 0].copy()
        image[:, :, 0] = image[:, :, 2].copy()
        image[:, :, 2] = tempo

        # convert to uint8
        image *= 255
        image[image > 255] = 255
        image[image < 0] = 0
        image = image.astype(np.uint8)

        return image

    def from_img_space_to_cv2(self, image):
        # for tanh
        # tranpose .. from (Channels, H, W) to (H, W, Channels)
        image = np.transpose(image, (1, 2, 0))
        # scale from [-1, 1] to [-0.5, 0.5]
        image *= 0.5
        # translate from [-0.5, 0.5] to [0.0, 1.0]
        image += 0.5

        # swap channels
        tempo = image[:, :, 0].copy()
        image[:, :, 0] = image[:, :, 2].copy()
        image[:, :, 2] = tempo

        # convert to uint8
        image *= 255
        # these should not be needed for Tanh
        image[image > 255] = 255
        image[image < 0] = 0
        image = image.astype(np.uint8)

        return image

    def from_img_space_to_cv2_scaled(self, image):
        # for tanh
        # tranpose .. from (Channels, H, W) to (H, W, Channels)
        image = np.transpose(image, (1, 2, 0))
        # scale from [-2.75, 2.75] to range unormalized range
        image[:, :, 0] *= 0.229
        image[:, :, 1] *= 0.224
        image[:, :, 2] *= 0.225
        # add mean color (should now be between 0.0 to 1.0)
        image[:, :, 0] += 0.485
        image[:, :, 1] += 0.456
        image[:, :, 2] += 0.406

        # swap channels
        tempo = image[:, :, 0].copy()
        image[:, :, 0] = image[:, :, 2].copy()
        image[:, :, 2] = tempo

        # convert to uint8
        image *= 255
        # these should not be needed for Tanh
        image[image > 255] = 255
        image[image < 0] = 0
        image = image.astype(np.uint8)

        return image

    def reconstruct(self, PIL_image):
        # assert isinstance(PIL_image, Image)
        batch_img = FCN_LectureNet.prepare_image(PIL_image)

        using_cuda = next(self.parameters()).is_cuda

        if using_cuda:
            batch_img = batch_img.cuda(0)

        with torch.no_grad():
            res = self.forward(batch_img)

        if using_cuda:
            res = res.cpu()

        # De-normalize ...
        image = res[0].numpy()

        # for tanh ...
        image = self.from_img_space_to_cv2(image)

        return image

    @staticmethod
    def prepare_image(PIL_image):
        img_t = TF.to_tensor(PIL_image)
        # ... normalize the RGB values
        # img_t = TF.normalize(img_t, [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # ... simply put RGB values in same range as Tanh
        img_t = TF.normalize(img_t, [0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        img_t = torch.unsqueeze(img_t, 0)

        return img_t

    @staticmethod
    def CreateFromConfig(config, in_channels, reconstruction_mode):
        n_convs_down_1 = config.get("FCN_BINARIZER_NET_DOWN_CONV_FILTERS_1", 16)
        n_convs_down_2 = config.get("FCN_BINARIZER_NET_DOWN_CONV_FILTERS_2", 32)
        n_convs_down_3 = config.get("FCN_BINARIZER_NET_DOWN_CONV_FILTERS_3", 64)
        n_convs_down_4 = config.get("FCN_BINARIZER_NET_DOWN_CONV_FILTERS_4", 128)
        n_convs_down_5 = config.get("FCN_BINARIZER_NET_DOWN_CONV_FILTERS_5", 256)

        n_convs_mid = config.get("FCN_BINARIZER_NET_MIDDLE_CONV_FILTERS_MIDDLE", 512)

        n_upscale_5 = config.get("FCN_BINARIZER_NET_UPSAMPLE_FILTERS_5", 256)
        n_convs_up_5 = config.get("FCN_BINARIZER_NET_UP_CONV_FILTERS_5", 256)

        n_upscale_4 = config.get("FCN_BINARIZER_NET_UPSAMPLE_FILTERS_4", 128)
        n_convs_up_4 = config.get("FCN_BINARIZER_NET_UP_CONV_FILTERS_4", 128)

        n_upscale_3 = config.get("FCN_BINARIZER_NET_UPSAMPLE_FILTERS_3", 64)
        n_convs_up_3 = config.get("FCN_BINARIZER_NET_UP_CONV_FILTERS_3", 64)

        n_upscale_2 = config.get("FCN_BINARIZER_NET_UPSAMPLE_FILTERS_2", 32)
        n_convs_up_2 = config.get("FCN_BINARIZER_NET_UP_CONV_FILTERS_2", 32)

        n_upscale_1 = config.get("FCN_BINARIZER_NET_UPSAMPLE_FILTERS_1", 16)
        n_convs_up_1 = config.get("FCN_BINARIZER_NET_UP_CONV_FILTERS_1", 16)

        n_pix_feats_1 = config.get("FCN_BINARIZER_NET_PIXEL_FEATURES_1", 32)
        n_pix_feats_2 = config.get("FCN_BINARIZER_NET_PIXEL_FEATURES_2", 16)

        pix_kernel_size = config.get("FCN_BINARIZER_NET_PIXEL_KERNEL_SIZE", 3)

        kernel_size = config.get("FCN_BINARIZER_NET_KERNEL_SIZE", 3)

        lecture_net = FCN_LectureNet(in_channels,
                                     n_convs_down_1, n_convs_down_2, n_convs_down_3, n_convs_down_4, n_convs_down_5,
                                     n_convs_mid,
                                     n_upscale_5, n_convs_up_5, n_upscale_4, n_convs_up_4, n_upscale_3, n_convs_up_3,
                                     n_upscale_2, n_convs_up_2, n_upscale_1, n_convs_up_1, kernel_size,
                                     n_pix_feats_1, n_pix_feats_2, pix_kernel_size, reconstruction_mode)

        return lecture_net
