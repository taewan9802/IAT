import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
import os
import argparse
import tqdm
from torchvision.models import vgg16, VGG16_Weights
from data_loaders.lol_v1_whole import lowlight_loader_new
from model.IlluminationAdaptiveTransformer import IAT
#from model.new_temp import IAT
from IQA_pytorch import SSIM
from utils import PSNR, adjust_learning_rate, validation, LossNetwork

def train(config):
    if not os.path.exists(config.snapshots_folder):
        os.makedirs(config.snapshots_folder)
    model = IAT().cuda()

    # if use pretrain weight #
    model.load_state_dict(torch.load(config.pretrain_dir))

    # Data Setting
    train_dataset = lowlight_loader_new(images_path=config.img_path)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=config.train_batch_size,
                                                shuffle=True,
                                                num_workers=config.num_worker,
                                                pin_memory=True)
    val_dataset = lowlight_loader_new(images_path=config.img_val_path, mode='test')
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=config.val_batch_size,
                                                shuffle=True,
                                                num_workers=config.num_worker,
                                                pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    # Loss & Optimizer Setting & Metric
    vgg_model = vgg16(pretrained=True).features[:16].cuda()
    #vgg_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].cuda()

    for param in vgg_model.parameters():
        param.requires_grad = False

    L1_loss = nn.L1Loss()
    L1_smooth_loss = F.smooth_l1_loss
    loss_network = LossNetwork(vgg_model).eval()

    ssim = SSIM()
    psnr = PSNR()
    ssim_high = 0
    psnr_high = 0
    ssim_epoch = 0
    psnr_epoch = 0

    print('######## Start IAT Training #########')
    for epoch in range(config.num_epochs):
        print('\nEpoch : ', epoch)
        #adjust_learning_rate(optimizer, epoch)
        total_loss=0
        num_iteration=0
        for iteration, imgs in enumerate(tqdm.tqdm(train_loader)):
            num_iteration = num_iteration+1
            low_img, high_img = imgs[0].cuda(), imgs[1].cuda()
            optimizer.zero_grad()
            model.train()
            mul, add, enhance_img = model(low_img)

            #loss = L1_loss(enhance_img, high_img)
            #loss = L1_smooth_loss(enhance_img, high_img)+0.04*loss_network(enhance_img, high_img)
            loss = F.huber_loss(enhance_img, high_img, reduction='mean',delta=1.0)
            total_loss = total_loss + loss
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Evaluation Model
        model.eval()
        torch.save(model.state_dict(), os.path.join(config.snapshots_folder, 'best_Epoch' + str(epoch) + '.pth'))
        print("The Loss Value =  %.6f" %(total_loss/num_iteration))
        SSIM_mean, PSNR_mean = validation(model, val_loader)
        with open(config.snapshots_folder + 'log.txt', 'a+') as f:
            f.write('Epoch%3d : SSIM = %.6f, PSNR = %.6f, Loss = %.6f\n' %((epoch), (float(SSIM_mean)), (float(PSNR_mean)), (float(total_loss)/num_iteration)))
        if SSIM_mean > ssim_high:
            ssim_high = SSIM_mean
            ssim_epoch = epoch
        if PSNR_mean > psnr_high:
            psnr_high = PSNR_mean
            psnr_epoch = epoch
        print(f'Epoch{ssim_epoch} the highest SSIM value = {str(ssim_high)}')
        print(f'Epoch{psnr_epoch} the highest PSNR value = {str(psnr_high)}')
        f.close()

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    parser = argparse.ArgumentParser()

    ## crop64 #
    #parser.add_argument('--img_path', type=str, default='dataset/newdataset/crop64/train/low/')
    #parser.add_argument('--img_val_path', type=str, default='dataset/newdataset/crop64/test/low/')
    #parser.add_argument('--snapshots_folder', type=str, default='weights/version3/crop64/')
    #parser.add_argument('--train_batch_size', type=int, default=32)
    #parser.add_argument('--val_batch_size', type=int, default=16)
    #parser.add_argument('--num_worker', type=int, default=12)
    #parser.add_argument('--lr', type=float, default=0.00002)
    #parser.add_argument('--weight_decay', type=float, default=0.00004)
    ##parser.add_argument('--pretrain_dir', type=str, default='weights/best_Epoch.pth')
    #parser.add_argument('--num_epochs', type=int, default=100)

    ## crop128 #
    #parser.add_argument('--img_path', type=str, default='dataset/newdataset/crop128/train/low/')
    #parser.add_argument('--img_val_path', type=str, default='dataset/newdataset/crop128/test/low/')
    #parser.add_argument('--snapshots_folder', type=str, default='weights/version3/crop128/')
    #parser.add_argument('--train_batch_size', type=int, default=16)
    #parser.add_argument('--val_batch_size', type=int, default=8)
    #parser.add_argument('--num_worker', type=int, default=12)
    #parser.add_argument('--lr', type=float, default=0.00002)
    #parser.add_argument('--weight_decay', type=float, default=0.00002)
    #parser.add_argument('--pretrain_dir', type=str, default='best_crop64_v3.pth')
    #parser.add_argument('--num_epochs', type=int, default=100)

    # crop256 #
    parser.add_argument('--img_path', type=str, default='dataset/newdataset/crop256/train/low/')
    parser.add_argument('--img_val_path', type=str, default='dataset/newdataset/crop256/test/low/')
    parser.add_argument('--snapshots_folder', type=str, default='weights/version3/crop256/')
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_worker', type=int, default=12)
    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--pretrain_dir', type=str, default='best_crop128_v3.pth')
    parser.add_argument('--num_epochs', type=int, default=200)

    config = parser.parse_args()
    print(config)
    train(config)
