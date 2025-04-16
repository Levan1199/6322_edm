import os 
import sys 
import glob 

if __name__ == "__main__":
    chpt_fldr ="ncsnpp"
    files = sorted(glob.glob(f"{chpt_fldr}/*.chkpt"))
    
    for idx, file in enumerate(files):
        if idx<3:
            continue
        # cmd = f'python /home/diffusion_edm/6322_diffusion/eval/generate.py --outdir=fid-tmp{idx} --seeds=0-10 --subdirs --network="/home/diffusion_edm/6322_diffusion/{file}"'
        cmd=f"python eval/fid.py calc --images=fid-tmp{idx}     --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz --num=11"
        os.system(cmd)
    breakpoint()
