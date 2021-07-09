import os

i = 0

for root, dirs, files in os.walk("/home/cse/phd/anz208849/scratch/saved_models/CCM/plabel/train/0/cityscapes"):
    for name in files:
        print(os.path.join(root, name))
        # print(name)
        # i = i+1   #595 images 
        # print(i)
        