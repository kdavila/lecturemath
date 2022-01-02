
import os
import sys
import time

import cv2

import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from PIL import Image

from AM_CommonTools.configuration.configuration import Configuration

from AccessMath.lecturenet_v1.FCN_lecturenet_dataset import LectureNet_DataSet
from AccessMath.lecturenet_v1.FCN_lecturenet import FCN_LectureNet
from AccessMath.lecturenet_v1.util import LectureNet_Util


def main():
    if len(sys.argv) < 2:
        print("Usage")
        print("\tpython {0:s} config".format(sys.argv[0]))
        print("With:")
        print("\tconfig\tPath to configuration file")
        return

    start_time = time.time()
    start_loading = time.time()

    # read the config file
    config = Configuration.from_file(sys.argv[1])

    use_cuda = config.get("FCN_BINARIZER_USE_CUDA", True)

    kfbin_crop_size = config.get("FCN_BINARIZER_TRAIN_CROP_SIZE", (255, 255))
    kfbin_flip_chance = config.get("FCN_BINARIZER_TRAIN_CROP_FLIP_CHANCE", 0.5)

    kfbin_color_change_chance = config.get("FCN_BINARIZER_TRAIN_COLOR_CHANGE_CHANCE", 0.5)
    kfbin_lum_change_chance = config.get("FCN_BINARIZER_TRAIN_LUMINOSITY_CHANGE_CHANCE", 0.5)
    kfbin_noise_chance = config.get("FCN_BINARIZER_TRAIN_GAUSSIAN_NOISE_CHANCE", 0.25)
    kfbin_noise_level = config.get("FCN_BINARIZER_TRAIN_GAUSSIAN_NOISE_LEVEL", 15.0)

    kfbin_weight_expansion = config.get_int("FCN_BINARIZER_TRAIN_WEIGHT_EXPANSION", 1)
    kfbin_weight_extra = config.get("FCN_BINARIZER_TRAIN_WEIGHT_FOREGROUND_EXTRA", 5.0)

    # get data location
    images_dir = config.get_str("FCN_BINARIZER_PRETRAIN_IMAGES_DIR")
    masks_dir = config.get_str("FCN_BINARIZER_PRETRAIN_MASKS_DIR")

    # get debug data location
    debug_dir = config.get_str("FCN_BINARIZER_PRETRAIN_TEXT_DEBUG_DIR")

    # this is good for small datasets that can be kept in memory
    pre_load_images = config.get("FCN_BINARIZER_PRETRAIN_PRELOAD_IMAGES", False)

    kfbin_batch_size = config.get("FCN_BINARIZER_PRETRAIN_BATCH_SIZE", 8)
    learning_rate = config.get("FCN_BINARIZER_PRETRAIN_LEARNING_RATE", 0.1)
    n_epochs = config.get("FCN_BINARIZER_PRETRAIN_EPOCHS", 25)

    rec_median = config.get("FCN_BINARIZER_PRETRAIN_REC_MEDIAN", False)
    rec_median_blur_k = config.get_int("FCN_BINARIZER_PRETRAIN_REC_MEDIAN_BLUR_K", 35)

    use_reconstruction_output = config.get("FCN_BINARIZER_PRETRAIN_USE_RECONSTRUCTION_OUTPUT", False)

    pretrained_network_filename = config.get_str("FCN_BINARIZER_PRETRAIN_TEXT_OUTPUT", "FCN_PRETRAINED_TEXT.dat")

    output_dir = config.get_str("OUTPUT_PATH")
    show_debug_images = config.get_bool("FCN_BINARIZER_PRETRAIN_TEXT_DEBUG_SAVE", False)
    debug_image_prefix = config.get_str("FCN_BINARIZER_PRETRAIN_TEXT_DEBUG_PREFIX", "DEBUG_TEXT_")

    full_debug_image_prefix = output_dir + "/" + debug_image_prefix
    full_pretrained_network_filename = output_dir + "/" + pretrained_network_filename

    all_image_paths, all_masks_paths = LectureNet_Util.get_images_w_masks_filenames(images_dir, masks_dir)

    print("A total of {0:d} images with masks were found".format(len(all_image_paths)))

    lecture_kf_dataset = LectureNet_DataSet(all_image_paths[:], all_masks_paths[:], False, crop_size=kfbin_crop_size,
                                            flip_chance=kfbin_flip_chance,
                                            color_change_chance=kfbin_color_change_chance,
                                            luminosity_changes_chance=kfbin_lum_change_chance,
                                            gaussian_noise_chance=kfbin_noise_chance,
                                            gaussian_noise_range=kfbin_noise_level,
                                            weight_expansion=kfbin_weight_expansion,
                                            weight_fg_extra=kfbin_weight_extra,
                                            text_region_masks_expansion=0,
                                            reconstruct_median=rec_median, reconstruct_median_K=rec_median_blur_k)

    if pre_load_images:
        print("Pre-loading training images")
        lecture_kf_dataset.preload()
    else:
        print("Images will not be pre-loaded!")

    train_loader = torch.utils.data.DataLoader(lecture_kf_dataset, batch_size=kfbin_batch_size, shuffle=True,
                                               num_workers=0)

    end_loading = time.time()

    start_training = time.time()

    lecture_net = FCN_LectureNet.CreateFromConfig(config, 3, use_reconstruction_output)

    if use_reconstruction_output:
        print("Will train a Network using model pre-trained for reconstruction")
        pretrained_rec_filename = config.get("FCN_BINARIZER_PRETRAIN_RECONSTRUCTION_OUTPUT")
        full_pretrained_rec_filename = output_dir + "/" + pretrained_rec_filename

        # Loading a pre-trained model for reconstruction
        lecture_net.load_state_dict(torch.load(full_pretrained_rec_filename))
        # set the network in evaluation mode to freeze the Batch-normalization layers in common trunk
        lecture_net.eval()

        # ...reset the output branches required here ....
        lecture_net.reset_main_branches(3, config)
        # ... disable reconstruction mode ....
        lecture_net.reconstruction_mode = False

    else:
        # creating (and training) a new model from scratch
        print("Will train a Network for Text Detection from Scratch")

    # clipping the norm for large gradients
    nn.utils.clip_grad_norm_(lecture_net.parameters(), 1.0)

    pytorch_total_params = sum(p.numel() for p in lecture_net.parameters() if p.requires_grad)
    print("Total Trainable Parameters in Network: " + str(pytorch_total_params))

    bce_mask_loss = nn.BCEWithLogitsLoss(reduction="mean")
    bce_binary_loss = nn.BCEWithLogitsLoss(reduction="mean")
    mse_loss = nn.MSELoss(reduction="mean")

    # move to cuda
    if use_cuda:
        lecture_net = lecture_net.cuda(0)
        bce_mask_loss = bce_mask_loss.cuda(0)
        bce_binary_loss = bce_binary_loss.cuda(0)
        mse_loss = mse_loss.cuda(0)

    optimizer = optim.SGD(lecture_net.parameters(), lr=learning_rate, momentum=0.0)

    if show_debug_images:
        # This is just for DEBUGGING
        tempo_paths = LectureNet_Util.get_only_images_filenames(debug_dir)
        debug_imgs = [Image.open(tempo_path) for tempo_path in tempo_paths]
    else:
        debug_imgs = []

    for epoch in range(n_epochs):
        print("Starting Epoch # " + str(epoch + 1))
        epoch_loss = 0.0
        for i, (images, labels, weights, text_mask, medians) in enumerate(train_loader, 0):
            print(f"{i + 1}/{len(train_loader)}", end="\r")

            # clean gradient!
            optimizer.zero_grad()

            # get the inputs
            if use_cuda:
                images = images.cuda(0)
                labels = labels.cuda(0)
                text_mask = text_mask.cuda(0)
                medians = medians.cuda(0)

            # NOTE: these no longer have sigmoid applied!!
            # ..... sigmoid will be required when USING regular BCE or MSE loss
            out_binary, out_text_mask, out_recons = lecture_net(images)

            # Train mask prediction using original ground truth (black = background, white=text)  ...
            mask_loss = bce_mask_loss(out_text_mask, labels)

            # Train main branch
            binary_loss = bce_binary_loss(out_binary, text_mask)

            # Train reconstruction branch
            rec_loss = mse_loss(out_recons, medians)

            loss = binary_loss + mask_loss + rec_loss

            loss.backward()
            optimizer.step()

            # print statistics
            epoch_loss += loss.item()

            if use_cuda:
                torch.cuda.empty_cache()

        epoch_loss /= (len(lecture_kf_dataset))
        print(" - Epoch Loss: " + str(epoch_loss))

        if epoch % 5 == 0:
            torch.save(lecture_net.state_dict(), f"{full_pretrained_network_filename}.epoch_{epoch + 1}.bak")

        for idx, img in enumerate(debug_imgs):
            binary, text_mask, rec_img = lecture_net.binarize(img, True)
            cv2.imwrite(f"{full_debug_image_prefix}_BIN_{idx}_{epoch + 1}.png", binary)
            cv2.imwrite(f"{full_debug_image_prefix}_REC_{idx}_{epoch + 1}.png", rec_img)
            cv2.imwrite(f"{full_debug_image_prefix}_MASK_{idx}_{epoch + 1}.png", text_mask)

    lecture_net.eval()
    torch.save(lecture_net.state_dict(), full_pretrained_network_filename)

    end_training = time.time()

    end_time = time.time()

    print("Total time loading: " + str(end_loading - start_loading))
    print("Total time training: " + str(end_training - start_training))
    print("Total time: " + str(end_time - start_time))


if __name__ == "__main__":
    main()
