from models.ncsn_pp import NCSNpp, VEPrecond
import torch

if __name__ == "__main__":
    net = NCSNpp()
    net_vp =VEPrecond()
    breakpoint()