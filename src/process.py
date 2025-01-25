import cv2
import numpy as np
import torch
from PIL import Image


def binarize_img_np(img):
    tensor = torch.tensor(img).to(torch.float32)
    tensor = tensor.sum(2, keepdim=True).repeat(1, 1, 3)
    zeros = torch.zeros_like(tensor)
    zeros[tensor != 0] = 255
    zeros = zeros.to(torch.uint8)
    return zeros.numpy()


def data_pipeline(img_path, sketch_path, mask_path):
    img = np.array(Image.open(img_path).convert('RGB'))
    sketch = np.array(Image.open(sketch_path).convert('RGB'))
    sketch = binarize_img_np(sketch)
    mask = np.array(Image.open(mask_path).convert('RGB'))
    mask = binarize_img_np(mask)

    img = cv2.resize(img, (256, 256),
                     interpolation=cv2.INTER_CUBIC).astype(np.float32)
    grey = img.mean(axis=2)[:, :, np.newaxis].astype(np.float32)
    grey = np.concatenate((grey, grey, grey), axis=2)
    sketch = cv2.resize(sketch, (256, 256),
                        interpolation=cv2.INTER_NEAREST).astype(np.float32)
    mask = cv2.resize(mask, (256, 256),
                      interpolation=cv2.INTER_NEAREST).astype(np.float32)

    img = (img / 255. - 0.5) / 0.5
    mask = mask / 255.0
    sketch = sketch / 255.0
    canvas = np.zeros_like(sketch) - 1.0
    canvas[sketch >= 0.6] = 1.0
    sketch = canvas
    grey = (grey / 255. - 0.5) / 0.5

    img = torch.from_numpy(img.transpose(2, 0, 1))
    mask = torch.from_numpy(mask.transpose(2, 0, 1))
    sketch = torch.from_numpy(sketch.transpose(2, 0, 1))
    grey = torch.from_numpy(grey.transpose(2, 0, 1))

    sample = {
        'img_src': img.to(torch.float32),
        'img_mask': mask.to(torch.float32),
        'img_edge': sketch.to(torch.float32),
        'img_grey': grey.to(torch.float32)
    }

    sample['img_mask'] = sample['img_mask'][0:1, :, :].unsqueeze(0).cuda()
    sample['img_edge'] = sample['img_edge'].unsqueeze(0).cuda()
    sample['img_src'] = sample['img_src'].unsqueeze(0).cuda()
    sample['img_grey'] = sample['img_grey'].unsqueeze(0).cuda()
    sample['img_mask'][sample['img_mask'] >= 0.5] = 1.0
    sample['img_mask'][sample['img_mask'] < 0.5] = 0.0

    return sample
