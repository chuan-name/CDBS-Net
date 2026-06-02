import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import argparse


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from network.CDBSNet import CDBSNet

import imageio
import torch.nn as nn
from utils.dataloader import test_dataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./checkpoints/CDBSNet-best/CDBSNet-best.pth')

opt = parser.parse_args()
model = Baseline()
model.load_state_dict(torch.load(opt.pth_path))
model.cuda()
model.eval()

for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    data_path = './dataset/TestDataset/{}'.format(_data_name)
    save_path = './results/CDBSNet-best/{}/'.format(_data_name)
    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        if gt.max() > 1:
            gt = (gt > 127).astype(np.float32)

        image = image.cuda()
        with torch.no_grad():

            res1 = model(image)[0]
            res1 = F.interpolate(res1, size=gt.shape, mode='bilinear', align_corners=False)

            res = res1.data.cpu().numpy().squeeze()


            res_mask = (res <= 0).astype(np.float32)

            imageio.imwrite(save_path + name, (res_mask * 255).astype(np.uint8))
