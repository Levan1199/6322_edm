import os 
import sys 
import glob 
import cv2 
import numpy as np 

if __name__ == "__main__":
    chpt_fldr ="ncsnpp"
    files = sorted(glob.glob(f"{chpt_fldr}/*.chkpt"))
    all_ims = list()
    for idx, file in enumerate(files):
        # if idx<3:
        #     continue
        # cmd = f'python /home/diffusion_edm/6322_diffusion/eval/generate.py --outdir=fid-tmp{idx} --seeds=0-10 --subdirs --network="/home/diffusion_edm/6322_diffusion/{file}"'
        # cmd=f"python eval/fid.py calc --images=fid-tmp{idx}     --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz --num=11"
        # os.system(cmd)
        ims = glob.glob(f"fid-tmp{idx}/**/*.png", recursive=True)
        print(ims)
        images =list()
        for im in ims:
            images.append(cv2.imread(im))
        
        all_ims.append(images)
    # all_ims_np = np.array(all_ims)
    res_img = np.zeros((640, 640, 3), dtype=np.uint8)
    step = len(all_ims)//20

    for i in range(20):
        # if st
        im_batch = all_ims[i*step]
        for j in range(10):
            res_img[i*32:i*32+32, j*32: j*32+32, :] = im_batch[j]
    cv2.imwrite('test.png', res_img)
    breakpoint()
