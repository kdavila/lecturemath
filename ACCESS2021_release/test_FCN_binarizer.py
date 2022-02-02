import sys

import torch

import cv2
from PIL import Image

from AM_CommonTools.configuration.configuration import Configuration

from AccessMath.lecturenet_v1.FCN_lecturenet import FCN_LectureNet
# from proto_FCN_lecturenet import FCN_LectureNet

def main():
    if len(sys.argv) < 5:
        print("Usage:")
        print("\tpython {0:s} config network input_img output_prefix".format(sys.argv[0]))
        print("Where:")
        print("\tconfig:\t\tPath to configuration used to train the network")
        print("\tnetwork:\tPath to trained network")
        print("\tinput_img\t:Path to Input Image to binarize")
        print("\toutput_prefix\t:Prefix of output images")
        return

    config_filename = sys.argv[1]

    # read the configuration file ....
    config = Configuration.from_file(sys.argv[1])

    use_cuda = config.get("FCN_BINARIZER_USE_CUDA", True)

    model_filename = sys.argv[2]
    input_filename = sys.argv[3]
    output_prefix = sys.argv[4]

    print("... loading model ...")
    # lecture_net = torch.load(model_filename)

    lecture_net = FCN_LectureNet.CreateFromConfig(config, 3, False)
    if use_cuda:
        lecture_net.load_state_dict(torch.load(model_filename))
    else:
        lecture_net.load_state_dict(torch.load(model_filename,map_location=torch.device('cpu')))
    lecture_net.eval()

    if use_cuda:
        lecture_net = lecture_net.cuda()

    pytorch_total_params = sum(p.numel() for p in lecture_net.parameters() if p.requires_grad)
    print("Total Trainable Parameters in Network: " + str(pytorch_total_params))

    # load test image ...
    raw_image = cv2.imread(input_filename)
    pil_image = Image.fromarray(cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR))

    binary, text_mask, rec_img = lecture_net.binarize(pil_image, return_others=True, force_binary=True)
    cv2.imwrite(output_prefix + "_BIN.png", binary)
    cv2.imwrite(output_prefix + "_text.png", text_mask)
    cv2.imwrite(output_prefix + "_bg.png", rec_img)
    print("data saved!")


if __name__ == '__main__':
    main()

