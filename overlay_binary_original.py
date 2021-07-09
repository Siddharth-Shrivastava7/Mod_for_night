import os 
from PIL import Image, ImageChops
from tqdm import tqdm
import numpy as np
# i = 0 

with open('/home/sidd_s/Mod_for_night/dataset/list/darkzurich/dz_val_rf_dannet.txt') as f: 
    content = f.readlines() 

pred_cols = [x.strip() for x in content if x.find('dannet_pred') != -1] 
pred_bin = '/home/sidd_s/scratch/data_hpc/data/dark_zurich_val/gt/rf_fake_dannet/'
src_path = '/home/sidd_s/scratch/data_hpc/data/dark_zurich_val/gt/'


for i in tqdm(range(len(pred_cols))): 
    nm = pred_cols[i].split('/')[-1].replace('rgb_anon_color', 'gt_labelColor')
    binary = Image.open(pred_bin + nm)
    binary = np.array(binary) 
    # changing the binary
    # binary[binary==40] = 127
    # binary[binary==0] = 4
    # binary[binary==255] = 0 
    # binary[binary==4] = 255
    # print(np.unique(binary)) # [  0  40 255]
    im = Image.fromarray(binary)
    im.save(pred_bin + nm)
    # print('yes')
    # break 

print('finished it')

# for i in tqdm(range(len(pred_cols))): 
#     pred = Image.open(src_path + pred_cols[i]).convert('RGB')
#     nm = pred_cols[i].split('/')[-1].replace('rgb_anon_color', 'gt_labelColor')
#     binary = Image.open(pred_bin + nm).convert('RGB') 
#     im3 = ImageChops.multiply(binary, pred) 
#     im3.save( src_path + 'gt_overlay_rf_pred_dannet_mod/' + nm)  

# print('done')


# for root, _, files in os.walk("color"):
#     for name in files:
#         i += 1
#         print(i) 
#         pathp = os.path.join(root,name)
#         # print(pathp)
#         # print(name) 
#         pred = Image.open(pathp).convert('RGB')
#         nameb = name.replace('rgb_anon', 'rf_pred')
#         pathb = os.path.join('rf_city_dzval', nameb)
#         binary = Image.open(pathb).convert('RGB')
#         im3 = ImageChops.multiply(binary, pred)
#         im3.save('overlay_rf_pred/' + name)  
