import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import os
import sys
import argparse
import random
from glob import glob
import cv2
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


from network.CDBSNet import CDBSNet

from utils.dataloader import PolypDataset
from utils.loss import DeepSupervisionLoss
from utils.utils import AvgMeter, clip_gradient
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils.eval_functions import Fmeasure_calu

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def eval_dice(model, val_loader, device):
    model.eval()
    dices = []
    with torch.no_grad():
        for images, gts in val_loader:
            images = images.to(device)
            gts = gts.to(device)
            preds = model(images)[0]

            preds = preds.cpu().numpy()
            gts = gts.cpu().numpy()
            for pred, gt in zip(preds, gts):
                pred = np.squeeze(pred)
                gt = np.squeeze(gt)
                if pred.shape != gt.shape:
                    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

                pred_mask = (pred <= 0).astype(np.float32)
                gt_mask = (gt > 0.5).astype(np.float32)
                _, _, _, dice, _, _ = Fmeasure_calu(pred_mask, gt_mask, 0.5)
                dices.append(dice)
    model.train()
    return np.mean(dices)


def train(train_loader, model, optimizer, epoch, criteria_loss):
    model.train()

    size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()
    best = 0
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()

            images, gts = pack
            images = Variable(images).to(device)
            gts = Variable(gts).to(device).float()
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

                gts = gts.float()
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='nearest')
            predicts = model(images)
            loss = criteria_loss(predicts, gts)

            writer.add_scalar("Loss/train", loss, epoch)

            loss.backward()
            clip_gradient(optimizer, opt.grad_norm)
            optimizer.step()

            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)


        torch.cuda.empty_cache()


        if i % 10 == 0:
            torch.cuda.empty_cache()

        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], ' 'loss: {:.4f}'.format(datetime.now(), epoch,
                                                                                           opt.epoch, i, total_step,
                                                                                           loss_record.show()))

    save_path = 'checkpoints/{}/'.format(opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)



if __name__ == '__main__':
    writer = SummaryWriter()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=80, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-3, help='learning rate')
    parser.add_argument('--grad_norm', type=float, default=0.5, help='gradient clipping norm')
    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--mt', type=float, default=0.9)
    parser.add_argument('--power', type=float, default=0.9)
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--train_path', type=str,
                        default='./dataset/TrainDataset', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='CDBSNet-best')
    parser.add_argument("--mgpu", type=str, default="false", choices=["true", "false"])
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
    opt = parser.parse_args()


    image_root = '{}/image/'.format(opt.train_path)
    gt_root = '{}/mask/'.format(opt.train_path)
    all_images = sorted(
        [f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')])

    random.seed(42)
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.9)
    train_list = all_images[:split_idx]
    val_list = all_images[split_idx:]



    def get_subset_loader(image_root, gt_root, file_list, batchsize, trainsize, shuffle, augmentation):
        images = [os.path.join(image_root, f) for f in file_list]
        gts = [os.path.join(gt_root, f) for f in file_list]
        dataset = PolypDatasetSubset(images, gts, trainsize, augmentation)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, num_workers=0,
                                             pin_memory=True)
        return loader


    class PolypDatasetSubset(PolypDataset):
        def __init__(self, images, gts, trainsize, augmentations):
            self.trainsize = trainsize
            self.augmentations = augmentations
            self.images = images
            self.gts = gts
            self.size = len(self.images)
            if self.augmentations == True:
                self.img_transform = transforms.Compose([
                    transforms.RandomRotation(90, interpolation=InterpolationMode.BILINEAR, expand=False, center=None),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.Resize((self.trainsize, self.trainsize)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                self.gt_transform = transforms.Compose([
                    transforms.RandomRotation(90, interpolation=InterpolationMode.NEAREST, expand=False, center=None),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.Resize((self.trainsize, self.trainsize)),
                    transforms.ToTensor()])
            else:
                self.img_transform = transforms.Compose([
                    transforms.Resize((self.trainsize, self.trainsize)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                self.gt_transform = transforms.Compose([
                    transforms.Resize((self.trainsize, self.trainsize)),
                    transforms.ToTensor()])


    train_loader = get_subset_loader(image_root, gt_root, train_list, opt.batchsize, opt.trainsize, shuffle=True,
                                     augmentation=True)
    val_loader = get_subset_loader(image_root, gt_root, val_list, opt.batchsize, opt.trainsize, shuffle=False,
                                   augmentation=False)
    total_step = len(train_loader)

    model = Baseline()
    if opt.mgpu == "true" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)


    set_seed(42)

    model.to(device)
    params = model.parameters()
    optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=opt.mt, weight_decay=opt.weight_decay)

    criteria_loss = DeepSupervisionLoss(typeloss="SDFEdgeLoss")

    start_epoch = 0
    if opt.resume is not None:
        if os.path.isfile(opt.resume):
            print(f"Loading checkpoint from {opt.resume}")
            checkpoint = torch.load(opt.resume, map_location=device)


            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)

            print("Checkpoint loaded successfully!")

            start_epoch = 50
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print(f"Checkpoint file {opt.resume} not found!")
            exit(1)

    lr_lambda = lambda epoch: 1.0 - pow((epoch / opt.epoch), opt.power)
    scheduler = LambdaLR(optimizer, lr_lambda)


    if opt.resume is not None and start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()

    print(torch.cuda.get_device_name(0))
    print("#" * 20, "Start Training", "#" * 20)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)


    if opt.resume is not None and start_epoch > 0:
        best_dice = 0.9297
        print(f"从检查点恢复，设置历史最佳Dice: {best_dice:.4f}")
    else:
        best_dice = 0

    save_path = 'checkpoints/{}/'.format(opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    init_val_dice = eval_dice(model, val_loader, device)
    print(f'初始模型验证集Dice: {init_val_dice:.4f}')

    for epoch in range(start_epoch, opt.epoch):
        train(train_loader, model, optimizer, epoch, criteria_loss)
        scheduler.step()
        val_dice = eval_dice(model, val_loader, device)
        print(f'Epoch {epoch + 1} 验证集Dice: {val_dice:.4f} (历史最佳: {best_dice:.4f})')
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), save_path + 'CDBSNet-best.pth')
            print('[保存最优模型]:', save_path + 'CDBSNet-best.pth')
    writer.close()
