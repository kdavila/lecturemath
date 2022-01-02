
import os
import sys
import time

import cv2

import torch
import torch.utils
import torch.utils.data
import torch.optim as optim
import torch.nn as nn

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

    kfbin_invert_color_chance = config.get("FCN_BINARIZER_TRAIN_COLOR_INVERT_CHANGE", 0.5)

    kfbin_color_change_chance = config.get("FCN_BINARIZER_TRAIN_COLOR_CHANGE_CHANCE", 0.5)
    kfbin_lum_change_chance = config.get("FCN_BINARIZER_TRAIN_LUMINOSITY_CHANGE_CHANCE", 0.5)
    kfbin_noise_chance = config.get("FCN_BINARIZER_TRAIN_GAUSSIAN_NOISE_CHANCE", 0.25)
    kfbin_noise_level = config.get("FCN_BINARIZER_TRAIN_GAUSSIAN_NOISE_LEVEL", 15.0)

    rec_median = config.get("FCN_BINARIZER_PRETRAIN_REC_MEDIAN", False)
    rec_median_blur_k = config.get_int("FCN_BINARIZER_PRETRAIN_REC_MEDIAN_BLUR_K", 35)

    # get data location
    images_dir = config.get_str("FCN_BINARIZER_PRETRAIN_REC_IMAGES_DIR")

    # get debug data location
    debug_dir = config.get_str("FCN_BINARIZER_PRETRAIN_REC_DEBUG_DIR", None)

    # this is good for small datasets that can be kept in memory
    pre_load_images = config.get("FCN_BINARIZER_PRETRAIN_PRELOAD_IMAGES", False)

    kfbin_batch_size = config.get("FCN_BINARIZER_PRETRAIN_BATCH_SIZE", 8)
    learning_rate = config.get("FCN_BINARIZER_PRETRAIN_REC_LEARNING_RATE", 0.1)
    n_epochs = config.get("FCN_BINARIZER_PRETRAIN_REC_EPOCHS", 25)

    show_debug_images = config.get_bool("FCN_BINARIZER_PRETRAIN_REC_DEBUG_SAVE", False)
    debug_image_prefix = config.get_str("FCN_BINARIZER_PRETRAIN_REC_DEBUG_PREFIX", "DEBUG_REC_")
    pretrained_network_filename = config.get_str("FCN_BINARIZER_PRETRAIN_RECONSTRUCTION_OUTPUT", "FCN_PRETRAINED_REC.dat")


    output_dir = config.get_str("OUTPUT_PATH")
    full_debug_image_prefix = output_dir + "/" + debug_image_prefix
    full_pretrained_network_filename = output_dir + "/" + pretrained_network_filename

    all_image_paths = LectureNet_Util.get_only_images_filenames(images_dir)

    print("A total of {0:d} images with masks were found".format(len(all_image_paths)))

    lecture_kf_dataset = LectureNet_DataSet(all_image_paths[:], None, True, crop_size=kfbin_crop_size,
                                            flip_chance=kfbin_flip_chance,
                                            color_invert_chance=kfbin_invert_color_chance,
                                            color_change_chance=kfbin_color_change_chance,
                                            luminosity_changes_chance=kfbin_lum_change_chance,
                                            gaussian_noise_chance=kfbin_noise_chance,
                                            gaussian_noise_range=kfbin_noise_level,
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

    lecture_net = FCN_LectureNet.CreateFromConfig(config, 3, True)
    if use_cuda:
        lecture_net = lecture_net.cuda(0)

    # clipping the norm for large gradients
    nn.utils.clip_grad_norm_(lecture_net.parameters(), 1.0)

    pytorch_total_params = sum(p.numel() for p in lecture_net.parameters() if p.requires_grad)
    print("Total Trainable Parameters in Network: " + str(pytorch_total_params))

    mse_loss = nn.MSELoss(reduction="mean")

    optimizer = optim.SGD(lecture_net.parameters(), lr=learning_rate, momentum=0.0)

    # for DEBUGGING
    if debug_dir is not None:
        tempo_paths = LectureNet_Util.get_only_images_filenames(debug_dir)
        debug_imgs = [Image.open(tempo_path) for tempo_path in tempo_paths]
    else:
        debug_imgs = []

    if show_debug_images:
        for idx, img in enumerate(debug_imgs):
            reconstructed = lecture_net.reconstruct(img)
            cv2.imwrite(f"{full_debug_image_prefix}_{idx}_0.png", reconstructed)

    for epoch in range(n_epochs):
        print("Starting Epoch # " + str(epoch + 1))
        epoch_loss = 0.0
        for i, (images, labels, weights, text_mask, medians) in enumerate(train_loader, 0):
            print(f"{i + 1}/{len(train_loader)}", end="\r")

            optimizer.zero_grad()

            # print("mini batch {0:d}".format(i), flush=True)
            # get the inputs
            if use_cuda:
                images = images.cuda(0)
                if rec_median:
                    # use medians as the target
                    labels = medians.cuda(0)
                else:
                    # use original image as target
                    labels = labels.cuda(0)

            out_reconstruction = lecture_net(images)

            # Train reconstruction branch
            # Labels should contain same image using normalization with mean=(0.5,0.5,0.5) and std=(0.5,0.5,0.5)
            # so all values should be between [-1, 1] as produced by Tanh activations
            # rec_loss = mse_loss(out_reconstruction, labels)

            # Case where we had re-scaled the Tanh output to produce values between -2.75 to 2.75
            rec_loss = mse_loss(out_reconstruction, labels)

            # loss = binary_loss + mask_loss
            loss = rec_loss

            loss.backward()
            optimizer.step()

            # print statistics
            epoch_loss += loss.item()

            if use_cuda:
                torch.cuda.empty_cache()

        # epoch_loss /= (len(lecture_kf_dataset) * kfbin_crop_size[0] * kfbin_crop_size[1])
        epoch_loss /= (len(lecture_kf_dataset))
        print(" - Epoch Loss: " + str(epoch_loss))

        if epoch % 2 == 0:
            torch.save(lecture_net.state_dict(), f"{full_pretrained_network_filename}.epoch_{epoch + 1}.bak")

        if show_debug_images:
            for idx, img in enumerate(debug_imgs):
                reconstructed = lecture_net.reconstruct(img)
                cv2.imwrite(f"{full_debug_image_prefix}_{idx}_{epoch + 1}.png", reconstructed)

    lecture_net.eval()
    torch.save(lecture_net.state_dict(), full_pretrained_network_filename)

    end_training = time.time()

    end_time = time.time()

    print("Total time loading: " + str(end_loading - start_loading))
    print("Total time training: " + str(end_training - start_training))
    print("Total time: " + str(end_time - start_time))


if __name__ == "__main__":
    main()
