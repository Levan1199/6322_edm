#! /usr/bin/bash
source /home/DRL_env/venv/bin/activate
latest_chkpts=($(ls -tr /home/diffusion_edm/6322_diffusion/ncsnpp | grep training-state))
latest_chkpt=${latest_chkpts[-1]}
echo "Using checkpoint $latest_chkpt to generate ims"
export PYTHONPATH=$PYTHONPATH:$(pwd)
python /home/diffusion_edm/6322_diffusion/eval/generate.py --outdir=fid-tmp --seeds=0-10 --subdirs \
    --network="/home/diffusion_edm/6322_diffusion/ncsnpp/$latest_chkpt"