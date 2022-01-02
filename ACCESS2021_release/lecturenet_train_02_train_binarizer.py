
import os
import sys
import time

import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from PIL import Image

from AM_CommonTools.configuration.configuration import Configuration
from AccessMath.data.meta_data_DB import MetaDataDB

from AccessMath.lecturenet_v1.FCN_lecturenet_dataset import LectureNet_DataSet
from AccessMath.lecturenet_v1.FCN_lecturenet import FCN_LectureNet
from AccessMath.lecturenet_v1.util import LectureNet_Util


def extract_kf_image_binary_annotation_pairs(root_dir, database, dataset_name):
    training_set = database.get_dataset(dataset_name)

    all_images_locations = []
    all_gt_locations = []
    for lecture in training_set:
        # print(lecture.title.lower())
        annotation_prefix = root_dir + "/" + database.output_annotations + "/" + database.name + "_" + lecture.title.lower()

        annot_image_dir = annotation_prefix + "/keyframes"
        annot_binary_dir = annotation_prefix + "/binary"

        lecture_image_elements = os.listdir(annot_image_dir)
        lecture_binary_elements = os.listdir(annot_binary_dir)

        for img_filename in lecture_image_elements:
            base, ext = os.path.splitext(img_filename)

            if ext.lower() == ".png":
                # keyframe image ... find in binary gt
                if img_filename in lecture_binary_elements:
                    # valid key-frame (has Ground Truth) ... add!
                    all_images_locations.append(annot_image_dir + "/" + img_filename)
                    all_gt_locations.append(annot_binary_dir + "/" + img_filename)

    return all_images_locations, all_gt_locations


def weighted_mse_loss(output, target, weights):
    return torch.mean(((output - target) ** 2) * weights)


def combined_per_pixel_mse_bce_loss(output, target, weights, per_pixel_bce, bce_weight):
    return torch.mean(((output - target) ** 2) * weights + per_pixel_bce * bce_weight)


def per_pixel_weighted_bce_loss(weights, per_pixel_bce):
    return torch.mean(per_pixel_bce * weights)


def main():
    if len(sys.argv) < 2:
        print("usage")
        print("\tpython {0:s} config".format(sys.argv[0]))
        print("\n\nwhere")
        print("\tconfig:\tAccessMath Configuration File")
        return

    start_time = time.time()

    start_loading = time.time()

    config = Configuration.from_file(sys.argv[1])

    # load the database
    try:
        database = MetaDataDB.from_file(config.get_str("VIDEO_DATABASE_PATH"))
    except Exception as e:
        print("Invalid database file")
        print(e)
        return

    output_dir = config.get_str("OUTPUT_PATH")

    use_cuda = config.get("FCN_BINARIZER_USE_CUDA", True)

    kfbin_crop_size = config.get("FCN_BINARIZER_TRAIN_CROP_SIZE", (255, 255))
    kfbin_crop_remove_empty_borders = config.get("FCN_BINARIZER_TRAIN_CROP_REMOVE_EMPTY_BORDERS", False)
    kfbin_crop_min_fg = config.get("FCN_BINARIZER_TRAIN_CROP_MIN_FOREGROUND", 0.05)
    kfbin_flip_chance = config.get("FCN_BINARIZER_TRAIN_CROP_FLIP_CHANCE", 0.5)

    kfbin_invert_color_chance = config.get("FCN_BINARIZER_TRAIN_COLOR_INVERT_CHANGE", 0.5)

    kfbin_color_change_chance = config.get("FCN_BINARIZER_TRAIN_COLOR_CHANGE_CHANCE", 0.5)
    kfbin_lum_change_chance = config.get("FCN_BINARIZER_TRAIN_LUMINOSITY_CHANGE_CHANCE", 0.5)
    kfbin_noise_chance = config.get("FCN_BINARIZER_TRAIN_GAUSSIAN_NOISE_CHANCE", 0.25)
    kfbin_noise_level = config.get("FCN_BINARIZER_TRAIN_GAUSSIAN_NOISE_LEVEL", 15.0)

    kfbin_weight_expansion = config.get_int("FCN_BINARIZER_TRAIN_WEIGHT_EXPANSION", 1)
    kfbin_weight_extra = config.get("FCN_BINARIZER_TRAIN_WEIGHT_FOREGROUND_EXTRA", 5.0)

    kfbin_text_masks_expansion = config.get_int("FCN_BINARIZER_TRAIN_TEXT_MASK_EXPANSION", 10)

    kfbin_batch_size = config.get("FCN_BINARIZER_TRAIN_BATCH_SIZE", 8)
    learning_rate = config.get("FCN_BINARIZER_TRAIN_LEARNING_RATE", 0.1)

    n_epochs = config.get("FCN_BINARIZER_TRAIN_EPOCHS", 25)
    bg_class_weight = config.get("FCN_BINARIZER_TRAIN_BG_CLASS_WEIGHT", None)

    use_pretraining_output = config.get("FCN_BINARIZER_TRAIN_USE_PRETRAIN_OUTPUT", False)
    pretrained_is_reconstruction = config.get("FCN_BINARIZER_TRAIN_FROM_RECONSTRUCTION_PRETRAIN", False)
    pretraining_output_path = config.get("FCN_BINARIZER_TRAIN_PRETRAIN_OUTPUT")

    trained_network_filename = config.get_str("FCN_BINARIZER_TRAIN_OUTPUT", "FCN_BIN_TRAINED.dat")

    show_debug_images = config.get_bool("FCN_BINARIZER_TRAIN_DEBUG_SAVE", False)
    debug_dir = config.get_str("FCN_BINARIZER_TRAIN_DEBUG_DIR")
    debug_image_prefix = config.get_str("FCN_BINARIZER_TRAIN_DEBUG_PREFIX", "DEBUG_TEXT_")

    full_debug_image_prefix = output_dir + "/" + debug_image_prefix
    full_trained_network_filename = output_dir + "/" + trained_network_filename

    image_locs, gt_locs = extract_kf_image_binary_annotation_pairs(output_dir, database, "training")

    print("A total of {0:d} training keyframes were found".format(len(image_locs)))


    # full res (1920 x 1080) -> crashes
    # half res (960 x 540) -> can handle batch size 2
    # quarter res (480 x 270) -> can handle batch size 12
    # eight res (240 x 135) -> can handle batch size 48
    # custom (256 x 256) -> can haandle batch size 20

    lecture_kf_dataset = LectureNet_DataSet(image_locs[:], gt_locs[:], False, crop_size=kfbin_crop_size,
                                            crop_remove_empty_borders=kfbin_crop_remove_empty_borders,
                                            crop_min_fg_prc=kfbin_crop_min_fg,
                                            flip_chance=kfbin_flip_chance,
                                            color_invert_chance=kfbin_invert_color_chance,
                                            color_change_chance=kfbin_color_change_chance,
                                            luminosity_changes_chance=kfbin_lum_change_chance,
                                            gaussian_noise_chance=kfbin_noise_chance,
                                            gaussian_noise_range=kfbin_noise_level,
                                            weight_expansion=kfbin_weight_expansion,
                                            weight_fg_extra=kfbin_weight_extra,
                                            text_region_masks_expansion=kfbin_text_masks_expansion)

    print("Pre-loading training images")
    lecture_kf_dataset.preload()

    print("Total Background pixels in Dataset: " + str(lecture_kf_dataset.total_background))
    print("Total Foreground pixels in Datasaet: " + str(lecture_kf_dataset.total_foreground))
    if bg_class_weight is None:
        bg_class_weight = lecture_kf_dataset.total_foreground / (lecture_kf_dataset.total_background + 0.001)

    train_loader = torch.utils.data.DataLoader(lecture_kf_dataset, batch_size=kfbin_batch_size, shuffle=True,
                                               num_workers=0)

    end_loading = time.time()

    start_training = time.time()

    if use_pretraining_output:
        full_pretrained_filename = output_dir + "/" + pretraining_output_path

        # Loading a pre-trained model
        lecture_net = FCN_LectureNet.CreateFromConfig(config, 3, pretrained_is_reconstruction)
        lecture_net.load_state_dict(torch.load(full_pretrained_filename))
        # freeze pre-trained weights for Batch Normalization
        lecture_net.eval()

        print("- Loaded: " + full_pretrained_filename)

        if pretrained_is_reconstruction:
            print("Will train a binarization model from pre-trained network for reconstruction")
            # ...reset the output branches required here ....
            lecture_net.reset_main_branches(3, config)
            # ... disable reconstruction mode ....
            lecture_net.reconstruction_mode = False
        else:
            # network is not on reconstruction mode and main branches need to be retrained, not reseted
            print("Will train a binarization model from pre-trained network for text detection")

    else:
        print("Will train a binarization model from scratch ....")

        lecture_net = FCN_LectureNet.CreateFromConfig(config, 3, False)


    # clipping the norm for large gradients
    nn.utils.clip_grad_norm_(lecture_net.parameters(), 1.0)

    pytorch_total_params = sum(p.numel() for p in lecture_net.parameters() if p.requires_grad)
    print("Total Trainable Parameters in Network: " + str(pytorch_total_params))

    # bce_loss = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=torch.as_tensor([bg_class_weight], dtype=torch.float))
    bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    bce_mask_loss = nn.BCEWithLogitsLoss(reduction="mean")

    # move to cuda
    if use_cuda:
        lecture_net = lecture_net.cuda(0)
        bce_loss = bce_loss.cuda(0)
        bce_mask_loss = bce_mask_loss.cuda(0)

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

            # print("mini batch {0:d}".format(i), flush=True)
            # get the inputs
            if use_cuda:
                images = images.cuda(0)
                labels = labels.cuda(0)
                # weights = weights.cuda(0)
                text_mask = text_mask.cuda(0)

            # NOTE: Main branch and Text Branch do not include a sigmoid layer at the end in order to be compatible
            #       with BCEWithLogits loss, but sigmoid will be required when USING regular BCE or MSE loss
            out_binary, out_text_mask, rec_img = lecture_net(images)

            # (BCE)
            binary_loss = bce_loss(out_binary, labels)
            mask_loss = bce_mask_loss(out_text_mask, text_mask)

            loss = binary_loss + mask_loss

            loss.backward()
            optimizer.step()

            # print statistics
            epoch_loss += loss.item()

            torch.cuda.empty_cache()

        epoch_loss /= (len(lecture_kf_dataset) * kfbin_crop_size[0] * kfbin_crop_size[1])
        print(" - Epoch Loss: " + str(epoch_loss))

        if epoch % 5 == 0:
            torch.save(lecture_net.state_dict(), f"{full_trained_network_filename}.epoch_{epoch + 1}.bak")

        for idx, img in enumerate(debug_imgs):
            binary, text_mask, rec_img = lecture_net.binarize(img, return_others=True, force_binary=False, apply_sigmoid=True)
            cv2.imwrite(f"{full_debug_image_prefix}_BIN_{idx:d}_{epoch + 1:d}.png", binary)
            cv2.imwrite(f"{full_debug_image_prefix}_REC_{idx:d}_{epoch + 1:d}.png", rec_img)
            cv2.imwrite(f"{full_debug_image_prefix}_MASK_{idx:d}_{epoch + 1:d}.png", text_mask)

    lecture_net.eval()
    torch.save(lecture_net.state_dict(), full_trained_network_filename)

    end_training = time.time()

    end_time = time.time()

    print("Total time loading: " + str(end_loading - start_loading))
    print("Total time training: " + str(end_training - start_training))
    print("Total time: " + str(end_time - start_time))


if __name__ == "__main__":
    main()
