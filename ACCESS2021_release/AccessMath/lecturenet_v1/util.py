
import os


class LectureNet_Util:
    @staticmethod
    def get_only_images_filenames(images_dir):
        all_image_paths = []
        all_elements = os.listdir(images_dir)
        for element in all_elements:
            base, ext = os.path.splitext(element)

            if ext.lower() in [".png", ".jpg"]:
                all_image_paths.append(images_dir + "/" + element)

        return all_image_paths

    @staticmethod
    def get_images_w_masks_filenames(images_dir, masks_dir):
        all_image_paths = []
        all_masks_paths = []
        all_elements = os.listdir(images_dir)
        for element in all_elements:
            base, ext = os.path.splitext(element)

            if ext.lower() in [".png", ".jpg"]:
                mask_path = masks_dir + "/" + base + ".png"
                if os.path.exists(mask_path):
                    all_image_paths.append(images_dir + "/" + element)
                    all_masks_paths.append(mask_path)

        return all_image_paths, all_masks_paths