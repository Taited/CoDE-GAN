import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import CODEGANBottleNeck, CODEGANEncoder, CODEGANGenerator


class CODEGAN(nn.Module):
    def __init__(self,
                 norm_edge=False,
                 input_sketch_mask=False,
                 is_valid_cr=True,
                 is_disc_cr=False):
        super().__init__()
        self.norm_edge = norm_edge
        self.input_sketch_mask = input_sketch_mask
        self.is_valid_cr = is_valid_cr
        self.is_disc_cr = is_disc_cr

        common_params = {
            'base_channels': 64,
            'pad_type': 'zero',
            'activation': 'lrelu',
            'norm': 'in',
            'init_type': 'xavier',
            'init_gain': 0.02
        }

        self.texture_encoder = CODEGANEncoder(in_channels=4,
                                              conv_type='GatedConv2d',
                                              **common_params)

        # Conditional structure encoder
        conv_type = 'GatedConv2d' if input_sketch_mask else 'Conv2d'
        self.structure_encoder = CODEGANEncoder(
            in_channels=3 if input_sketch_mask else 2,
            conv_type=conv_type,
            **common_params)

        self.bottle_neck = CODEGANBottleNeck(conv_type='GatedConv2d',
                                             **common_params)

        self.generator = CODEGANGenerator(out_channels=3,
                                          conv_type='GatedConv2d',
                                          **common_params)

    def forward(self, data_batch):
        img_mask = data_batch['img_mask']
        if img_mask.shape[1] > 1:
            img_mask = img_mask[:, 0:1, :, :]
        img_src = img_mask + data_batch['img_src'] * (1 - img_mask)
        img_grey = data_batch['img_grey'][:, 0:1, :, :]

        if self.norm_edge:
            img_edge = data_batch['img_edge'][:, 0:1, :, :]
            img_edge[img_mask == 0] = -1.
        else:
            img_edge = data_batch['img_edge'][:, 0:1, :, :] * img_mask

        if self.input_sketch_mask:
            img_edge = torch.concat((img_edge, img_mask), dim=1)

        src_feat = self.texture_encoder(img_src, img_mask)
        edge_feat = self.structure_encoder(
            torch.concat((img_grey * (1 - img_mask), img_edge), dim=1))
        bottle_feat = self.bottle_neck(src_feat, edge_feat)
        pred_list = self.generator(bottle_feat)
        pred = pred_list[-1]

        img_fake = data_batch['img_src'] * (
            1 - data_batch['img_mask']) + pred * data_batch['img_mask']
        results = {key: data_batch[key] for key in data_batch}
        results['img_fake'] = img_fake
        results['img_masked'] = data_batch['img_src'] * (
            1 - data_batch['img_mask'])

        for i in range(len(pred_list) - 1):
            content_response = F.interpolate(pred_list[i],
                                             data_batch['img_mask'].shape[2:])
            results[f'unmasked_img_scale_{i}'] = content_response
            if self.is_valid_cr:
                results[f'img_scale_{i}'] = content_response
            else:
                # masking
                img_scale_tmp = (img_grey + 1) / 2 * (1 - img_mask)
                img_scale_tmp += content_response * img_mask
                results[f'img_scale_{i}'] = img_scale_tmp

        return results
