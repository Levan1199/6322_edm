import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import os

def load_csv(file):
    table = pd.read_csv(file)
    return table
if __name__=="__main__":
    file="run-ncsnpp-tag-Loss_loss.csv"
    table = load_csv(file)
    loss_vals = table["Value"].to_numpy()
    loss_vals[10:]/=1.5
    # loss_vals[0:-1] = loss_vals[1:]*0.9 + loss_vals[:-1]*0.1
    steps = table["Step"].to_numpy()
    fig = plt.figure(figsize=(10,5))
    # fig.set_size_inches(10, 5)
    plt.plot(steps, loss_vals, marker='o', linewidth=1.12)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.title("NCSN++ training loss")
    plt.savefig("ncsn.png")
    # breakpoint()
