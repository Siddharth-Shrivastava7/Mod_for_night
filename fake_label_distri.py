import os 
from PIL import Image 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

with open('dataset/list/darkzurich/val.txt') as f: 
    data = f.read().splitlines()

# print(data) 
lst = []
for i in range(19):
    lst.append(0)


### now seeing real distribution.....
for img in tqdm(data): 
    img_gt = np.array(Image.open(os.path.join('/home/sidd_s/scratch/data_hpc/data/dark_zurich_val/gt', img + '_gt_labelTrainIds.png')))
    name = img.split('/')[-1]
    img_rf_gt = np.array(Image.open(os.path.join('/home/sidd_s/scratch/data_hpc/data/dark_zurich_val/gt/rf_fake_dannet', name+'_gt_labelColor.png')))
    indfk = np.argwhere(img_rf_gt== 0)  ## for real 127..for fake 0..#both overall
    lenfk = len(indfk)
    img_gtfk = img_gt[indfk[:,0], indfk[:,1]]
    for i in range(19):
        lst[i] = lst[i] + (len(np.argwhere(img_gtfk==i)) / lenfk)

fig = plt.figure(figsize = (20, 5)) 
x = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'light', 'sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motocycle', 'bicycle']
y = lst 
plt.bar(x,y, width=0.4,color ='blue')
plt.xlabel('classes')
plt.ylabel('fake_ratio_gt')
plt.savefig('fake_ratio_dzval_gt.png')
print(lst)
     

    

     
    
    

    
        
    

