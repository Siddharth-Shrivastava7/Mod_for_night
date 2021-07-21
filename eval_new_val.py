import os, sys
import argparse
import numpy as np
import torch
from model.DeeplabV2 import *#Res_Deeplab
from model.Unet import * 
from collections import OrderedDict

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
from trainer.btad_trainer import *
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
    parser.add_argument("--dataset", type=str, default='darkzurich_val')
    parser.add_argument("--single", action='store_true')
    parser.add_argument("--model", default='deeplab')
    return parser.parse_args()

def print_iou(iou, acc, miou, macc):
    for ind_class in range(iou.shape[0]):
        print('===> {0:2d} : {1:.2%} {2:.2%}'.format(ind_class, iou[ind_class, 0].item(), acc[ind_class, 0].item()))
    print('mIoU: {:.2%} mAcc : {:.2%} '.format(miou, macc))

def compute_iou(model, testloader, args):
    model = model.eval()

    interp_backp = nn.UpsamplingNearest2d(size=(1080, 1920)) 
    interp = nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True)   # dark_zurich -> (1080,1920)
    union = torch.zeros(args.num_classes, 1,dtype=torch.float).cuda().float()
    inter = torch.zeros(args.num_classes, 1, dtype=torch.float).cuda().float()
    preds = torch.zeros(args.num_classes, 1, dtype=torch.float).cuda().float()
    gts = torch.zeros(args.num_classes, 1, dtype=torch.float).cuda().float()
    # 2nd best 
    # totalp = 0
    # Tp = 0 
    with torch.enable_grad():
        for index, batch in tqdm(enumerate(testloader)):
            image, label, edge, _, name = batch
#            edge = F.interpolate(edge.unsqueeze(0), (512, 1024)).view(1,512,1024)``
            # print(name)
            ## use when testing dark zurich val...
            # if name[0].find('dannet_pred')==-1:  
            #     continue
            ## 
            # print(image.shape)
            image.requires_grad = True  ## for test time gradient calculation
            output =  model(image.cuda())
            label = label.cuda()
            # print(output.shape) # torch.Size([1, 1, 512, 512]) 
            # print(output.type) 
            # print('label shape:{} output shape:{}'.format(label.shape, output.shape))
            output = interp(output).squeeze()  #org .. orginial size evalutation 
            # output = F.softmax(output, dim=0) # one exp... not worked
            # print(output.shape) # torch.Size([1080, 1920])
            # output = output.squeeze()
            # print(output.shape)
            # save_pred(output, './save/dark_zurich_val/btad', args.dataset +str(index)+'.png') # org
            # print(name[0])
            name =name[0].split('/')[-1]
            # save_pred(output, '../scratch/data/acdc_gt/tensor_val_pred_aug', name)  # current org 

            # C, H, W = output.shape # original
            # print(torch.unique(output))
            # print(torch.unique(torch.argmax(output, dim = 0)))

            H, W = output.shape  # focal loss is used / bce 
            # # print(H, W)
            C = 2

            ###########loss at test time  calc .. back prop to get which pixels have the highest gradients...visualisation process only i think 
            # @torch.enable_grad()   # ensure grads in possible no grad context for testing 
            loss = WeightedFocalLoss(alpha= 0.75, gamma=3)  ## loss which was used at the training time 
            # print(output.shape) # torch.Size([1080, 1920])
            # print(label.squeeze().shape) # torch.Size([1080, 1920])
            # print(image.shape) # torch.Size([1, 19, 512, 512])
            # print(image.requires_grad)
            # print(image.is_leaf) # true
            # print(output.is_leaf) # False
            # print(label.is_leaf) # true
            seg_loss = loss(output, label.squeeze().float()) 
            loss = seg_loss
            # print(seg_loss) 
            # print('******')
            loss.backward() 
            # print(image.grad) # image gradient while doing the backprop 
            # print(image.grad.shape) ##  Same as the original image (#torch.Size([1, 19, 512, 512]) 
            # print(torch.argmax(image.grad , dim =1).shape) 
            # pred_label_backp =  torch.argmax(image.grad, dim =1).unsqueeze(dim=1)   
            # print(pred_label_backp.shape) # torch.Size([1, 1, 512, 512]) 
            # pred_label_backp = interp_backp(pred_label_backp.to(torch.float32)).squeeze().cpu().numpy()  # by upsampling...not working...next plan
            # pred_label_backp =  torch.argmax(image.grad, dim =1) 
            # pred_label_backp = image.grad 
            imgg = image.grad.squeeze(dim=0)[0] # 1 channel # torch.Size([512, 512]) 
            # print(imgg.shape)
            imgg = F.softmax(imgg) # between 0 and 1 proba distribution torch.Size([512, 512]) 
            # print(np.unique(imgg))
            # print(imgg.shape)
            # imgg = np.array(imgg)*255 
            imgg = np.array(imgg)
            imgg_min = imgg.min()
            imgg_max = imgg.max()
            imgg_mean = np.mean(imgg)
            imgg_std = np.std(imgg)
            imgg = (imgg - imgg_min) / (imgg_max - imgg_min) ## min max normalisation 
            # imgg = imgg / imgg_max  ## only max normalisation
            # imgg = (imgg - imgg_mean) / (imgg_std) ## z score normalisation
            print(np.unique(imgg))
            # heatmap = cv2.applyColorMap(np.uint8(imgg*255), cv2.COLORMAP_HOT)
            # cv2.imwrite('heatmap.png',heatmap)
            # pred_label_backp = pred_label_backp.squeeze().cpu().numpy()
            # label_img_color = label_img_to_color(pred_label_backp)
            # im = Image.fromarray(label_img_color) 
            # im.save(os.path.join('../scratch/data_hpc/data/dark_zurich_val/gt/dz_val_pred_backprop', name))

            ## image pred from input..for testing 
            # img_pred = torch.argmax(image, dim=1).unsqueeze(dim=1)
            # img_pred = interp_backp(img_pred.to(torch.float32)).squeeze().detach().cpu().numpy()
            # img_pred = torch.argmax(image, dim=1)
            # img_pred = img_pred.squeeze().detach().cpu().numpy()
            # label_img_color = label_img_to_color(img_pred)
            # im = Image.fromarray(label_img_color) 
            # im.save(os.path.join('../scratch/data_hpc/data/dark_zurich_val/gt/dz_val_pred_backprop', name))
            # print(pred_label_backp.shape)  # torch.Size([1080, 1920])
            # print(pred_label_backp)
            # print(index)
            
            ############
            #########################################################################original
            # Mask = (label.squeeze())<C
            # # print(sum(sum(Mask==0)))
            # # print(Mask.shape) # torch.Size([1080, 1920])
            # # break
            # pred_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
            # pred_e = pred_e.repeat(1, H, W).cuda()
            # # pred = output.argmax(dim=0).float() # org 
            # # print(pred.shape)

            # #mod focal loss is used / bce 
            # pred = output.float()
            # pred[output>=0.5] = 1
            # pred[output<0.5] = 0
            # #mod

            # pred_mask = torch.eq(pred_e, pred).byte()
            # pred_mask = pred_mask*Mask

            # label_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
            # label_e = label_e.repeat(1, H, W).cuda()
            # label = label.view(1, H, W)
            # label_mask = torch.eq(label_e, label.float()).byte()
            # label_mask = label_mask*Mask

            # tmp_inter = label_mask+pred_mask

            # # calc for RR, RF, FR, FF 
            # # # print(tmp_inter.shape) #torch.Size([2, 1080, 1920])
            # # # print(label_mask.shape) #torch.Size([2, 1080, 1920])
            # # # print(torch.unique(tmp_inter)) # tensor([0, 1, 2], device='cuda:0', dtype=torch.uint8)
            # # # print(torch.unique(label_mask)) # tensor([0, 1], device='cuda:0', dtype=torch.uint8)
            # # # where tmp_inter equal to 2 there...is RR..where label_mask is 0 and pred_mask is 1 is...FR..similarly others 
            # # label = label.squeeze(dim = 0).cpu().numpy()
            # # pred = pred.cpu().numpy()
            # # # print(np.unique(pred)) # [0. 1.]
            # # # print(np.unique(label)) # [  0   1 255]
            # # # print(label.shape) # (1080, 1920)
            # # # print(pred.shape) # (1080, 1920)
            # # # print(tmp_inter_np.shape) # (2, 1080, 1920)
            # # # RR 
            # # indrr = np.argwhere((label==1) & (pred==1)) 
            # # # # indrr = np.where((label==1) & (pred==1))
            # # # # print('yahhoo')
            # # # # print('yes!!') 
            # # # indrr = np.transpose(indrr)
            # # # # print(indrr)
            # # # # FF  
            # # indff = np.argwhere((label==0) & (pred==0))
            # # # # RF 
            # # indrf = np.argwhere((label==1) & (pred==0))
            # # # # FR 
            # # indfr = np.argwhere((label==0) & (pred==1))
            # # # ignore 
            # # ign = np.argwhere(label==255)
            # # # print(len(list(indfr)) + len(list(indrr)) + len(list(indff)) + len(list(indrf)) + len(list(ign)))
            # # # print('*******')
            # # # print(1920*1080) 
            # # save_confusion_pred(pred, indrr, indrf, indff, indfr, ign, '../scratch/data/pred_dannet_rf/unet_foc_aug_resize_rf', name)
            # # # # # print('yes')
            # # # # break

            # cu_inter = (tmp_inter==2).view(C, -1).sum(dim=1, keepdim=True).float()
            # cu_union = (tmp_inter>0).view(C, -1).sum(dim=1, keepdim=True).float()
            # cu_preds = pred_mask.view(C, -1).sum(dim=1, keepdim=True).float()
            # cu_gts = label_mask.view(C, -1).sum(dim=1, keepdim=True).float()
            # # # print(cu_gts) # total number of labels of class fake and real in GT
            # # print('*******')
            # # print(cu_inter) # numbers
            # # break

            # union+=cu_union
            # inter+=cu_inter
            # preds+=cu_preds
            # gts+=cu_gts
            #########################################################################origninal

            ########################################################################with2ndbest
            # Mask = (label.squeeze())<C
            # pred_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
            # pred_e = pred_e.repeat(1, H, W).cuda()
            # pred = output.argmax(dim=0).float()  # most confident pred 
            # pred2 = output.topk(2,dim=0)[1][1].float()  # 2nd most confi   
            # pred_mask = torch.eq(pred_e, pred).byte()
            # pred2_mask = torch.eq(pred_e, pred2).byte()
            # pred_mask = pred_mask*Mask
            # pred2_mask = pred2_mask*Mask

            # label_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
            # label_e = label_e.repeat(1, H, W).cuda()
            # label = label.view(1, H, W)
            # label_mask = torch.eq(label_e, label.float()).byte()
            # label_mask = label_mask*Mask

            # pred_maskb = torch.logical_or(pred_mask, pred2_mask) # for combining both 1st most and 2 most confident pred       

            # tmp_interb = label_mask+pred_maskb  # for getting top 2 acc 
            # tmp_inter = label_mask + pred_mask  
            # tmp_inter2 = label_mask + pred2_mask   

            # cu_inter = (tmp_inter2==2).view(C, -1).sum(dim=1, keepdim=True).float() # overall including both predictions

            # #####ignore this 
            # cu_union = (tmp_inter2>0).view(C, -1).sum(dim=1, keepdim=True).float() 
            # cu_preds = pred2_mask.view(C, -1).sum(dim=1, keepdim=True).float() 
            # ##### ignore this 

            # Tp += sum((tmp_interb==2).view(C, -1).sum(dim=1, keepdim=True)).float()
            # totalp += H*W      

            # # print(pred_mask.shape)
            # union+=cu_union
            # inter+=cu_inter
            # preds+=cu_preds
            ##########################################################################with2ndbest
        
        ###########original
        iou = inter/union
        acc = inter/preds
        acc_gt = inter/gts
        mIoU = iou.mean().item()
        mAcc = acc.mean().item()
        mAcc_gt = acc_gt.mean().item()
        # print_iou(iou, acc, mIoU, mAcc) # org
        print_iou(iou, acc_gt, mIoU, mAcc_gt)
        ##################
        
        # print('*****************************')
        # mean_acc = Tp/totalp 
        # print('mean_acc: ', mean_acc)
        # print('*****************************')

        # iou = inter/union
        # acc = inter/preds
        # mIoU = iou.mean().item()
        # mAcc = acc.mean().item()
        # print_iou(iou, acc, mIoU, mAcc)    

        return iou, mIoU, acc, mAcc

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
        19: [0,  0, 0]
        }
    # # with open('./dataset/cityscapes_list/info.json') as f:
    #     data = json.load(f)

    # label_to_color = {
    #     0: [0, 0, 0],
    #     1: [255,255,255]
    # }
    # org for confusion mat
    # label_to_color = { 
    #     0: [0,0,0], # ignored region
    #     1: [0,0,255], # RR # Blue
    #     2: [0,255,0], # FF # Green 
    #     3: [255, 0, 0], #RF Red   
    #     4: [255,255,255] #FR White
    # }
    # label_to_color = {
    #     0: [0,0,0], # ignored region
    #     1: [127,127,127], # RR # Grey
    #     2: [255,255,255], # FF # white 
    #     3: [255,255,255], #RF white  
    #     4: [127,127,127] #FR Grey
    # }
    img_height, img_width = img.shape
    img_color = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    for row in range(img_height):
        for col in range(img_width):
            label = img[row][col]
            img_color[row, col] = np.array(label_to_color[label])
            # img_color[row][col] = np.asarray(data['palette'][label])
    return img_color


def save_fake(model, testloader):
    model = model.eval()
    # interp = nn.Upsample(size=(1024,2048), mode='bilinear', align_corners=True)
    with torch.no_grad():
        for index, batch in tqdm(enumerate(testloader)):
            image, label, edge, _, name = batch
            output =  model(image.cuda())
            # output = interp(output).squeeze()
            output = output.squeeze()
            name = name[0].split("/")[-1]
            # print(name)
            save_pred(output, '../scratch/saved_models/CCM/save/result/rf_city_dzval', name)
    return


def save_pred(pred, direc, name):
    # palette = get_palette(256)

    # tebn
    # torch.save(pred, osp.join(direc,name))

    # original >>>>>>
    pred = pred.cpu().numpy()
    # print(pred.shape)

    pred = np.asarray(np.argmax(pred, axis=0), dtype=np.uint8)   ##### original
    # pred = np.asarray(np.argsort(pred, axis= 0)[-2], dtype = np.uint8)  ############ 2nd best prediction
    
    # # if thresholding for binary segmentation  # org for one channel pred
    # pred[pred<0.5] = 0
    # pred[pred>=0.5] = 1
    
    # pred = np.asarray(np.argmax(pred, axis=0)) # org  for ce loss
    # print(pred.shape)
    label_img_color = label_img_to_color(pred)
    # print(label_img_color.dtype)
    # print(label_img_color.shape)
    # print(np.unique(label_img_color))
    # cv2.imwrite(osp.join(direc,name), label_img_color)
    im = Image.fromarray(label_img_color) # use always PIL or other library .. try to avoid cv2.. but if no other option then ok
    im.save(osp.join(direc,name))
    # original >>>>>> 
    return 
    # print('img saved!')
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

def save_confusion_pred(pred, indrr, indrf, indff, indfr, ign, direc, name):
    # pred = pred.cpu().numpy() 
    # pred = np.asarray(np.argmax(pred, axis=0))
    # print(pred.shape) # (1080, 1920)
    pred[ign[:,0], ign[:,1]] =  0 
    # org 
    pred[indrr[:,0], indrr[:,1]] =  1
    pred[indff[:,0], indff[:,1]] =  2
    pred[indrf[:,0], indrf[:,1]] =  3
    pred[indfr[:,0], indfr[:,1]] =  4
    # print(np.unique(pred))
    label_img_color = label_img_to_color(pred) 
    im = Image.fromarray(label_img_color) 
    im.save(osp.join(direc,name))
    return

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    args = get_arguments()
    with open('./config/so_configmodbtad_2.yml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = edict(cfg)
    cfg.num_classes=args.num_classes
    if args.single:
        #from model.fuse_deeplabv2 import Res_Deeplab
        if args.model=='deeplab':
            model = Res_Deeplab(num_classes=args.num_classes).cuda()
        elif args.model == 'unet':
            model = UNet(n_class=2).cuda()
        elif args.model == 'unet_gan':
            model = Unet_Discriminator(resolution = 256).cuda()
        elif args.model == 'unet_mod': 
            model = UNet_mod(n_class=2)
        elif args.model == 'unet_mod_2':
            model = UNet_mod_2(n_class=2)
        else:
            model = FCN8s(num_classes = args.num_classes).cuda() 

        # model = nn.DataParallel(model)
        model.load_state_dict(torch.load(args.frm)) # org
        # model.load_state_dict(torch.load(args.frm,strict=False))

        # original saved file with DataParallel
        # state_dict = torch.load(args.frm)
        # # create new OrderedDict that does not contain `module.`
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:] # remove `module.`
        #     new_state_dict[name] = v
        # # load params
        # model.load_state_dict(new_state_dict)

        model.eval().cuda()
        testloader  = init_test_dataset(cfg, args.dataset, set='val')
        # print('****************************************************')
        # save_fake(model, testloader)
        iou, mIoU, acc, mAcc = compute_iou(model, testloader, args) # original
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