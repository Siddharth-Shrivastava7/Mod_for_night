import os, sys
import argparse
import numpy as np
import torch
from model.DeeplabV2 import *#Res_Deeplab

from torch.utils import data
import torch.nn as nn
import os.path as osp
import yaml
# from utils.logger import Logger 
from dataset.dataset import *
from easydict import EasyDict as edict
from tqdm import tqdm
from PIL import Image
import json
import torchvision
import cv2


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--frm", type=str, default=None)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--dataset", type=str, default='cityscapes')
    parser.add_argument("--single", action='store_true')
    parser.add_argument("--model", default='deeplab')
    return parser.parse_args()

# def print_iou(iou, acc, miou, macc):
#     for ind_class in range(iou.shape[0]):
#         print('===> {0:2d} : {1:.2%} {2:.2%}'.format(ind_class, iou[ind_class, 0].item(), acc[ind_class, 0].item()))
#     print('mIoU: {:.2%} mAcc : {:.2%} '.format(miou, macc))

# def compute_iou(model, testloader, args):
#     model = model.eval()

#     interp = nn.Upsample(size=(1024,2048), mode='bilinear', align_corners=True)   # dark_zurich -> (1080,1920)
#     union = torch.zeros(args.num_classes, 1,dtype=torch.float).cuda().float()
#     inter = torch.zeros(args.num_classes, 1, dtype=torch.float).cuda().float()
#     preds = torch.zeros(args.num_classes, 1, dtype=torch.float).cuda().float()
#     with torch.no_grad():
#         for index, batch in tqdm(enumerate(testloader)):
#             image, label, edge, _, name = batch
#             # print(name)
# #            edge = F.interpolate(edge.unsqueeze(0), (512, 1024)).view(1,512,1024)
#             output =  model(image.cuda())
#             label = label.cuda()
#             # print('label shape:{} output shape:{}'.format(label.shape, output.shape))
#             output = interp(output).squeeze()
#             # save_pred(output, './save/cityscapes_val', args.dataset +str(index)+'.png') #always save the file according to name of input 
#             save_pred(output, './save/cityscapes_val/dark', name[0].split('/')[1])
#             C, H, W = output.shape
#             # print(torch.unique(output))
#             # print(torch.unique(torch.argmax(output, dim = 0)))

#             Mask = (label.squeeze())<C
#             pred_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
#             pred_e = pred_e.repeat(1, H, W).cuda()
#             pred = output.argmax(dim=0).float()
#             pred_mask = torch.eq(pred_e, pred).byte()
#             pred_mask = pred_mask*Mask

#             label_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
#             label_e = label_e.repeat(1, H, W).cuda()
#             label = label.view(1, H, W)
#             label_mask = torch.eq(label_e, label.float()).byte()
#             label_mask = label_mask*Mask

#             tmp_inter = label_mask+pred_mask
#             cu_inter = (tmp_inter==2).view(C, -1).sum(dim=1, keepdim=True).float()
#             cu_union = (tmp_inter>0).view(C, -1).sum(dim=1, keepdim=True).float()
#             cu_preds = pred_mask.view(C, -1).sum(dim=1, keepdim=True).float()

#             union+=cu_union
#             inter+=cu_inter
#             preds+=cu_preds

#         iou = inter/union
#         acc = inter/preds
#         mIoU = iou.mean().item()
#         mAcc = acc.mean().item()
#         print_iou(iou, acc, mIoU, mAcc)
#         return iou, mIoU, acc, mAcc


def label_img_to_color(img):
    label_to_color = {
        0: [128, 64,128],
        1: [244, 35,232],
        2: [ 70, 70, 70],
        3: [102,102,156],
        4: [190,153,153],
        5: [153,153,153],
        6: [250,170, 30],
        7: [220,220,  0],
        8: [107,142, 35],
        9: [152,251,152],
        10: [ 70,130,180],
        11: [220, 20, 60],
        12: [255,  0,  0],
        13: [  0,  0,142],
        14: [  0,  0, 70],
        15: [  0, 60,100],
        16: [  0, 80,100],
        17: [  0,  0,230],
        18: [119, 11, 32],
        19: [0,  0, 0],
        255: [255,255,255]
        }
    # with open('./dataset/cityscapes_list/info.json') as f:
    #     data = json.load(f)

    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]

            img_color[row, col] = np.array(label_to_color[label])
            # img_color[row, col] = np.asarray(data['palette'][label])

    return img_color


def save_pred(pred, savep):
    # palette = get_palette(256)
    # pred = pred.cpu().numpy()
    # pred = np.asarray(np.argmax(pred, axis=0), dtype=np.uint8)
    label_img_color = label_img_to_color(pred)
    # cv2.imwrite(osp.join(direc,name), label_img_color)
    im = Image.fromarray(label_img_color)
    im.save(savep)

    # img = np.zeros((pred.shape[0],pred.shape[1],3))

    # print(np.unique(pred))
    # print(pred.shape)
    # print(plabel.shape)
    # print(img.shape)
    # cv2.imwrite("pred.png", img)
    # output_im = Image.fromarray(pred)
    # output_im.putpalette(palette)
    # output_im.save('pred.png')
         
    # print((plabel==i).nonzero()) #work 
    # print(plabel==i)

    # img = torch.tensor(img)
    # img = img.permute(2, 0, 1).unsqueeze(dim = 0)
    # print(img.shape) 
    # torchvision.utils.save_image(img,osp.join(direc,name)) 

    # img = img.reshape(3,pred.shape[0],-1)
    # print(img.shape)
    # print(np.unique(img))
    # unique, counts = np.unique(img, return_counts=True)
    # print(dict(zip(unique, counts)))
    # img = Image.fromarray((img * 255).astype(np.uint8))
    # img.save(osp.join(direc,name))
                


def main():
    args = get_arguments()
    with open('./config/so_configmodbtad.yml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = edict(cfg)
    cfg.num_classes=args.num_classes
    # if args.single:
    #     #from model.fuse_deeplabv2 import Res_Deeplab
    #     if args.model=='deeplab':
    #         model = Res_Deeplab(num_classes=args.num_classes).cuda()
    #     else:
    #         model = FCN8s(num_classes = args.num_classes).cuda() 

    # cityscapes positive source effective pixels 
    # save_path = '/home/cse/phd/anz208849/scratch/saved_models/CCM/plabel/train/cityscapes_color_org_best' 
    # for root, dirs, files in os.walk("/home/cse/phd/anz208849/scratch/saved_models/CCM/plabel/train/0/cityscapes"):
    #     for name in tqdm(files):
    #         # print(os.path.join(root, name))
    #         load = os.path.join(root, name) 
    #         img = np.asarray(Image.open(load), dtype=np.uint8)
    #         savep = os.path.join(save_path, name)
    #         save_pred(img, savep)   


    # dz will do this
    img_ids =[]
    list_path = './dataset/list/darkzurich/train.txt'
    with open(list_path) as f:
        for item in f.readlines():
            fields = item.strip().split('\t')[0]
            if ' ' in fields:
                fields = fields.split(' ')[0]
            img_ids.append(fields)

    save_path = '../scratch/saved_models/CCM/plabel/train/0_col'
    load_path = '../scratch/saved_models/CCM/plabel/train/0'

    for name in tqdm(img_ids):
        img_name = name.split("/")[-1] + '_gt_labelIds.png'
        savep = os.path.join(save_path, img_name)
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)
        img_path = name + '_gt_labelIds.png'
        load = os.path.join(load_path, img_path)
        img = np.asarray(Image.open(load), dtype=np.uint8)            
        save_pred(img, savep)        

        # model = nn.DataParallel(model)
        # model.load_state_dict(torch.load(args.frm))
        # model.load_state_dict(torch.load(args.frm,strict=False))
        # model.eval().cuda()

        # testloader = init_test_datasetbtad(cfg, args.dataset, set='val')
        # iou, mIoU, acc, mAcc = compute_iou(model, testloader, args)
    return

    # sys.stdout = Logger(osp.join(cfg['result'], args.frm+'.txt'))

    # best_miou = 0.0
    # best_iter = 0
    # best_iou = np.zeros((args.num_classes, 1))

   
    # for i in range(args.start, 25):
    #     model_path = osp.join(cfg['snapshot'], args.frm, 'GTA5_{0:d}.pth'.format(i*2000))# './snapshots/GTA2Cityscapes/source_only/GTA5_{0:d}.pth'.format(i*2000)
    #     model = Res_Deeplab(num_classes=args.num_classes)
    #     #model = nn.DataParallel(model)

    #     model.load_state_dict(torch.load(model_path))
    #     model.eval().cuda()
    #     testloader = init_test_dataset(cfg, args.dataset, set='train') 

    #     iou, mIoU, acc, mAcc = compute_iou(model, testloader)

    #     print('Iter {}  finished, mIoU is {:.2%}'.format(i*2000, mIoU))

if __name__ == '__main__':
    main()
