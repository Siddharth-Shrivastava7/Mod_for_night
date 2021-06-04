import torch
from PIL import Image
import numpy as np
import scipy.misc
import os
import random

def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    return fft_amp, fft_pha

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src

def FDA_source_to_target_np( src_img, trg_img, L=0.1 ):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img #.cpu().numpy()
    trg_img_np = trg_img #.cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg


# single source and target
# im_src = Image.open('frankfurt_000000_000294_leftImg8bit.png').convert('RGB')
# im_trg = Image.open('GOPR0356_frame_000321_rgb_anon.png').convert('RGB')

# im_src = im_src.resize( (1024,512), Image.BICUBIC )
# im_trg = im_trg.resize( (1024,512), Image.BICUBIC )

# im_src = np.asarray(im_src, np.float32)
# im_trg = np.asarray(im_trg, np.float32)

# im_src = im_src.transpose((2, 0, 1))
# im_trg = im_trg.transpose((2, 0, 1))

# src_in_trg = FDA_source_to_target_np( im_src, im_trg, L=0.01 )

# src_in_trg = src_in_trg.transpose((1,2,0))
# scipy.misc.toimage(src_in_trg, cmin=0.0, cmax=255.0).save('dark_city.png') 

# directory 
with open("./dataset/list/cityscapes/val.txt") as f:
    content = f.readlines()
city = [x.strip() for x in content]

with open("./dataset/list/darkzurich/train.txt") as f:
    content = f.readlines()
darkzurich = [x.strip() for x in content]

# print(darkzurich[2416])

# print(len(city))
# print(len(darkzurich))

src_path = '../scratch/data/cityscapes/leftImg8bit/val'
pred_path = '../scratch/data/dark_zurich/rgb_anon'
save_path = '../scratch/data/cityscapes/leftImg8bit/val/dark_city'

for i in range(len(city)): 
    print(i)

    ind = random.randint(0,2415)

    name = city[i].split('/')[-1]
    im_src = Image.open(os.path.join(src_path,city[i])).convert('RGB')
    im_trg = Image.open(os.path.join(pred_path, darkzurich[ind] + '_rgb_anon.png')).convert('RGB')

    im_src = im_src.resize( (1024,512), Image.BICUBIC )
    im_trg = im_trg.resize( (1024,512), Image.BICUBIC )

    im_src = np.asarray(im_src, np.float32)
    im_trg = np.asarray(im_trg, np.float32)

    im_src = im_src.transpose((2, 0, 1))
    im_trg = im_trg.transpose((2, 0, 1))

    src_in_trg = FDA_source_to_target_np( im_src, im_trg, L=0.01 )

    src_in_trg = src_in_trg.transpose((1,2,0))
    scipy.misc.toimage(src_in_trg, cmin=0.0, cmax=255.0).save(os.path.join(save_path,name)) 

print('yo')


