
import os
import sys
import json

import numpy as np
import cv2

def main():
    if len(sys.argv) < 4:
        print("Usage:")
        print("\tpython {0:s} gt_json img_dir out_dir".format(sys.argv[0]))
        print("With")
        print("\tgt_json\tPath to ground truth location (JSON file)")
        print("\timg_dir\tPath to input image dir")
        print("\tout_dir\tPath to output mask dir")
        return

    json_filename = sys.argv[1]
    img_dir = sys.argv[2]
    out_dir = sys.argv[3]

    with open(json_filename, "r", encoding="utf-8") as in_file:
        all_gt = json.load(in_file)

    print("The dataset has a total of {0:d} images".format(len(all_gt)))

    count_small = 0
    for img_id in all_gt:
        print("Processing: " + img_id)
        img_gt = all_gt[img_id]

        # load image ...
        img_filename = img_dir + "/" + img_id + ".jpg"
        img = cv2.imread(img_filename)

        if img.shape[0] < 256 or img.shape[1] < 256:
            count_small += 1
            print((img_id, img.shape))

        # create empty output mask
        out_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        for text_region in img_gt:
            polygon = text_region["points"]
            out_mask = cv2.fillPoly(out_mask, [np.array(polygon).astype(np.int32)], (255,))

        out_filename = out_dir + "/" + img_id + ".png"
        cv2.imwrite(out_filename, out_mask)
        
    print("A total of {0:d} small images were detected!".format(count_small))

if __name__ == "__main__":
    main()

