import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import imageio
class BaseDataSet(data.Dataset):
    def __init__(self, root, list_path,dataset, num_class,  joint_transform=None, transform=None, label_transform = None, max_iters=None, ignore_label=255, set='val', plabel_path=None, max_prop=None, selected=None,centroid=None, wei_path=None):
        
        self.root = root
        self.list_path = list_path
        self.ignore_label = ignore_label
        self.set = set
        self.dataset = dataset
        self.transform = transform
        self.joint_transform = joint_transform
        self.label_transform = label_transform
        self.plabel_path = plabel_path
        self.centroid = centroid

        # if self.set !='train':
        #     self.list_path = (self.list_path).replace('train', self.set)

        self.img_ids =[]
        if selected is not None:
            self.img_ids = selected
        else:
            with open(self.list_path) as f:
                for item in f.readlines():
                    fields = item.strip().split('\t')[0]
                    if ' ' in fields:
                        fields = fields.split(' ')[0]
                    self.img_ids.append(fields)

        if not max_iters==None:
            # print(len(self.img_ids))
            # self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids))) #org
            # print(len(self.img_ids))
            # print(len(self.img_ids))
            pass

        # elif max_prop is not None:
        #     total = len(self.img_ids)
        #     to_sel = int(np.floor(total * max_prop))
        #     index = list( np.random.choice(total, to_sel, replace=False) )
        #     self.img_ids = [self.img_ids[i] for i in index]

        self.files = []
        self.id2train = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                          19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                          26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        if self.dataset =='synthia':
            imageio.plugins.freeimage.download()

        if dataset=='gta5':
            if self.plabel_path is None:
                label_root = osp.join(self.root, 'labels')
            else:
                label_root = self.plabel_path

            for name in self.img_ids:
                img_file = osp.join(self.root, "images/%s" % name)
                label_file = osp.join(label_root, "%s" % name)
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name
                })

        elif dataset=='cityscapes' or dataset=='cityscapes_val':
            if self.plabel_path is None:
                label_root = osp.join(self.root, 'gtFine', self.set)
            else:
                label_root = self.plabel_path 
            for name in self.img_ids:
                # print(name)
                img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
                label_name = name.replace('leftImg8bit', 'gtFine_labelIds')
                label_file =osp.join(label_root, '%s' % (label_name))
                self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name
                })
                # print(img_file)
                # print(label_file)

        elif dataset=='rf_city' or dataset=='rf_city_val':
            # print('hi')
            if self.plabel_path is None:
                label_root = ''
            else:
                label_root = self.plabel_path 
            for name in self.img_ids:
                # print(name)
                img_file = osp.join(self.root, "gtFine/%s/%s" % (self.set, name))
                # print(img_file)
                # label_name = name.replace('leftImg8bit', 'gtFine_labelIds')
                label_file = []
                self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name
                })
                # print(img_file)
                # print(label_file)

        elif dataset == 'darkzurich_val':
            if self.plabel_path is None:
                label_root = osp.join(self.root, 'gt')
            else:
                label_root = self.plabel_path 
            for name in self.img_ids:
                # print(name)
                img_file = osp.join(self.root, "rgb_anon/%s_rgb_anon.png" % (name))
                label_file =osp.join(label_root, '%s_gt_labelIds.png' % (name))

                # print(img_file)
                # print(label_file)

                self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name
                })
        
        elif dataset == 'darkzurich_val_rf':
            if self.plabel_path is None:
                label_root = osp.join(self.root, 'gt')
            else:
                label_root = self.plabel_path 
            for name in self.img_ids:
                # print(name)
                # img_file = osp.join(self.root, "rgb_anon/%s_rgb_anon.png" % (name))
                img_file =osp.join(label_root, '%s_gt_labelColor.png' % (name))
                label_file = []
                # print(img_file)
                # print(label_file)

                self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name
                })

        elif dataset == 'darkzurich':
            if self.plabel_path is None:
                label_root = ''
            else:
                label_root = self.plabel_path 
            for name in self.img_ids:
                img_file = osp.join(self.root, "rgb_anon/%s_rgb_anon.png" % (name))
                if self.plabel_path is None:
                    label_file = []
                else:
                    label_file =osp.join(label_root, '%s_gt_labelIds.png' % (name))               
                self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name
                })

        elif dataset=='night_city':
            if self.plabel_path is None:
                label_root = self.root 
            else:
                label_root = self.plabel_path 
            for name in self.img_ids:
                img_file = osp.join(self.root, "%s" % (name))

                label_name = name.replace('_leftImg8bit', '_gtCoarse_labelIds')
                label_name = label_name.replace('leftImg8bit', 'gtCoarse_daytime_trainvaltest')
                label_file =osp.join(label_root, '%s' % (label_name))

                # print(img_file)
                # print(label_file)
                # print(name)

                self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name
                }) 
        elif dataset == 'acdc':
            # print('yo')
            for name in self.img_ids:
                # print(name)
                img_file = osp.join(self.root,  name)
                tup = (('acdc_trainval','acdc_gt'),('_rgb_anon.png','_gt_labelIds.png')) 
                for r in tup: 
                    lbname = name.replace(*r)
                lbname = lbname.replace('rgb_anon','gt') 
                label_file = osp.join(self.root, lbname)
                self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name
                })
                # print(label_file)
                # print(name)
                # print(img_file)
                # break
                  

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        try:
            image = Image.open(datafiles["img"]).convert('RGB')
            # print(image.size)
            # img_ignore_mask = np.all(image!=(0,0,0), axis = -1)
            # mask = img_ignore_mask.astype(int)*255  
            # print(image.size)
            # image = image.resize((1024,512)) #(width, height) 
            # print(image.size)
            if self.dataset == 'rf_city' or 'rf_city_val':
                # print('hi')
                im = np.array(image).shape
                if 'gta' in datafiles["name"]:
                    # print('hi')
                    label = np.zeros((im[0], im[1]), dtype = np.long) #fake label # for cross entropy
                    # label = np.zeros((im[0], im[1]), dtype = np.uint8) #fake label 
                # print('yo')
                else:
                    im_arr = np.array(image)
                    indices = np.where(np.all(im_arr == (0,0,0), axis=-1))
                    black_inx = np.transpose(indices)
                    label = np.ones((im[0], im[1]), dtype = np.long) # real label # for cross entropy
                    label[black_inx[:,0], black_inx[:,1]] = 255
                    # label = np.ones((im[0], im[1]), dtype = np.uint8) # real label

                label = Image.fromarray(label.astype(np.uint8))
                # print('***************')
                # print(label.size)
                if self.joint_transform is not None:
                    image, label = self.joint_transform(image, label, None)
                if self.label_transform is not None:
                    label = self.label_transform(label)

            elif self.dataset == 'darkzurich_val_rf':
                im = np.array(image).shape
                label = np.ones((im[0], im[1]), dtype = np.uint8) #real
                label = Image.fromarray(label.astype(np.uint8))
                
            elif self.dataset == 'darkzurich' and self.plabel_path is None: # trg no gt labels
                label = []
            else:
                label = Image.open(datafiles["label"])
                label = np.asarray(label, np.uint8)
                label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
                # print(datafiles['label']) # error is for dz pseudo labels
                if self.plabel_path is None:
                    for k, v in self.id2train.items():
                        label_copy[label == k] = v
                else:
                    label_copy = label
                label = Image.fromarray(label_copy.astype(np.uint8))
                if self.joint_transform is not None:
                    image, label = self.joint_transform(image, label, None)
                if self.label_transform is not None:
                    label = self.label_transform(label)

            name = datafiles["name"]            
            
            if self.transform is not None:
                image = self.transform(image)
           
        except Exception as e:
            # print('hi')
            print(index)
            index = index - 1 if index > 0 else index + 1
            return self.__getitem__(index)
        return image, label, 0, 0, name


