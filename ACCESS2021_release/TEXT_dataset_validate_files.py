
import os
import sys

import cv2

from PIL import Image, ImageOps

def main():
    if len(sys.argv) < 2:
        print("Usage")
        print("\tpython {0:s} img_dir".format(sys.argv[0]))
        return

    image_dir = sys.argv[1]

    image_filenames = os.listdir(image_dir)

    with_issues = []
    count_changed = 0
    for filename in image_filenames:
        img_path = image_dir + "/" + filename

        changed = False

        # load images
        pil_image = Image.open(img_path)
        o_w, o_h = pil_image.size
        try:
            pil_image = ImageOps.exif_transpose(pil_image)
        except:
            # count this image
            with_issues.append(img_path)
            count_changed += 1
            # and do not process it further
            continue

        n_w, n_h = pil_image.size

        if pil_image.mode == "CMYK" or pil_image.mode == "L":
            pil_image = pil_image.convert('RGB')
            changed = True

        # print(pil_image.mode)

        if o_w != n_w:
            changed = True

        if o_w < 256 or o_h < 256:
            changed = True

        if changed:
            count_changed += 1
            with_issues.append(img_path)

    if len(with_issues) > 0:
        print("\n\nImages with issues: {0:d}".format(count_changed))
        print("List of images with issues")
        for img_name in with_issues:
            print(img_name)

if __name__ == "__main__":
    main()

