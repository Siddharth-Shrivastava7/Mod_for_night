import os 
from PIL import Image, ImageChops
from tqdm import tqdm
# i = 0 

with open('/home/cse/phd/anz208849/Mod_for_night/dataset/list/darkzurich/dz_val_rf_dannet.txt') as f: 
    content = f.readlines() 

pred_cols = [x.strip() for x in content if x.find('dannet_pred') != -1] 
pred_bin = '/home/cse/phd/anz208849/scratch/data/dark_zurich_val/gt/acdc_model_dannet/val_pred/'
src_path = '/home/cse/phd/anz208849/scratch/data/dark_zurich_val/gt/'

for i in tqdm(range(len(pred_cols))): 
    pred = Image.open(src_path + pred_cols[i]).convert('RGB')
    nm = pred_cols[i].split('/')[-1]
    binary = Image.open(pred_bin + nm).convert('RGB') 
    im3 = ImageChops.multiply(binary, pred) 
    im3.save( src_path + 'overlay_rf_pred_dannet/' + nm)  
    
print('done')

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
