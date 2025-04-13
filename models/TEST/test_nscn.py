from models.ncsn_pp import NSCNpp, VEPrecond
import torch

if __name__ == "__main__":
    net = NSCNpp()
    net_vp =VEPrecond()
    breakpoint()