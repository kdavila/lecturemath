
import os
import sys
import json
import shutil

import numpy as np
import cv2

def main():
    if len(sys.argv) < 9:
        print("Usage")
        print("\tpython {0:s} in_img_dir in_JSON_gt train_img_dir train_mask_dir valid_img_dir valid_mask_dir test_img_dir test_mask_dir".format(sys.argv[0]))
        print("Where:")
        print("\tin_img_dir:\tOriginal COCO image directory")
        print("\tin_JSON_gt:\tOriginal COCO TEXT ground truth in JSON Format")
        print("\ttrain_img_dir:\tOutput directory for Train COCO images with Text Annotations")
        print("\ttrain_mask_dir:\tOutput directory for Training Text Masks")
        print("\tvalid_img_dir:\tOutput directory for Train COCO images with Text Annotations")
        print("\tvalid_mask_dir:\tOutput directory for Training Text Masks")
        print("\ttest_img_dir:\tOutput directory for Train COCO images with Text Annotations")
        print("\ttest_mask_dir:\tOutput directory for Training Text Masks")
        return

    input_image_dir = sys.argv[1]
    input_json_gt_filename = sys.argv[2]
    train_image_dir = sys.argv[3]
    train_mask_dir = sys.argv[4]
    valid_image_dir = sys.argv[5]
    valid_mask_dir = sys.argv[6]
    test_image_dir = sys.argv[7]
    test_mask_dir = sys.argv[8]

    # read ground truth ....
    with open(input_json_gt_filename, "r") as in_file:
        full_gt = json.load(in_file)

    image_filename_template = "{0:s}/COCO_train2014_{1:s}.{2:s}"

    count_per_set = {}
    total_per_count = {}
    many_regions_images = []
    for idx, img_id in enumerate(full_gt['imgToAnns']):
        image_set = full_gt['imgs'][img_id]['set']
        if image_set in count_per_set:
            count_per_set[image_set] += 1
        else:
            count_per_set[image_set] = 1

        source_filename = image_filename_template.format(input_image_dir, img_id.zfill(12), "jpg")
        print("Processing: " + source_filename)

        if os.path.exists(source_filename):
            img_annotation_ids = full_gt['imgToAnns'][img_id]
            # list of numbers as ints ...

            if len(img_annotation_ids) > 50:
                many_regions_images.append(source_filename)

            # count images by number of text regions
            if len(img_annotation_ids) in total_per_count:
                total_per_count[len(img_annotation_ids)] += 1
            else:
                total_per_count[len(img_annotation_ids)] = 1

            # generate the mask ...
            img = cv2.imread(source_filename)
            h, w, _ = img.shape

            text_mask = np.zeros((h, w), dtype=np.uint8)
            for region_id in img_annotation_ids:
                region_annot = full_gt['anns'][str(region_id)]
                polygon = region_annot['polygon']
                # round and convert to integer
                polygon = np.array(polygon).round(0).astype(np.int32)
                # make into Pairs of 2D points
                polygon = polygon.reshape((int(polygon.shape[0] / 2), 2))
                # make into poly-line
                array_poly = [polygon]
                text_mask = cv2.fillPoly(text_mask, array_poly, (255,))

            # save image on destination
            if image_set.lower() == "train":
                destination_filename = image_filename_template.format(train_image_dir, img_id.zfill(12), "jpg")
                mask_filename = image_filename_template.format(train_mask_dir, img_id.zfill(12), "png")
            elif image_set.lower() == "val":
                destination_filename = image_filename_template.format(valid_image_dir, img_id.zfill(12), "jpg")
                mask_filename = image_filename_template.format(valid_mask_dir, img_id.zfill(12), "png")
            elif image_set.lower() == "test":
                destination_filename = image_filename_template.format(test_image_dir, img_id.zfill(12), "jpg")
                mask_filename = image_filename_template.format(test_mask_dir, img_id.zfill(12), "png")
            else:
                # an image that is not on any of the sets? ... ignore!
                continue

            # ... use copy function ... to avoid re-encoding the image
            shutil.copy(source_filename, destination_filename)
            # save mask on destination
            cv2.imwrite(mask_filename, text_mask)
        else:
            print("Warning: File not found: " + source_filename)

    print("\nTotal Images by count")
    for count_key in sorted(list(total_per_count.keys())):
        print("Total images with {0:d} text regions: {1:d}".format(count_key, total_per_count[count_key]))

    print("\tTotal Images per Set")
    for image_set in count_per_set:
        print("Total images in {0:s} set: {1:d}".format(image_set, count_per_set[image_set]))

    """
    for image_filename in sorted(many_regions_images):
        print(image_filename)
    """

if __name__ == "__main__":
    main()

