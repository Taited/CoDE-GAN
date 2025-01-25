from argparse import ArgumentParser

import torch
from torchvision.utils import save_image

from src import CODEGAN, data_pipeline


def arg_parse():
    parser = ArgumentParser()
    parser.add_argument('--img', type=str, required=True)
    parser.add_argument('--sketch', type=str, required=True)
    parser.add_argument('--mask', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    return parser.parse_args()


def main():
    args = arg_parse()
    sample = data_pipeline(args.img, args.sketch, args.mask)
    gan = CODEGAN().cuda()
    gan.eval()
    gan.load_state_dict(
        torch.load(args.ckpt, map_location='cpu', weights_only=True))
    with torch.no_grad():
        output = gan(sample)
    output = output['img_fake']
    save_image(output, args.save_path, normalize=True, value_range=(-1, 1))


if __name__ == '__main__':
    main()
