import numpy as np
data = [527.567,
        399.694,
        373.722,
        404.33,
        392.956,
        371.334,
        361.206,
        337.022,
        320.8, 
        312.739,
        323.345,
        325.206,
        319.878,
        313.405,
        313.598,
        387.163,
        299.263,
        297.325,
        300.174,
        292.87,
        301.441,
        309.441,
        309.084,
        308.74,
        308.46,
        310,
        388.415,
        304.851,
        307.627,
        308.323,
        200.029,
        292.716,
        298.657,
        313.687,
        301.938,
        316.535,
        321.053,
        385.23,
        296.047,
        306.615,
        310.407,
        303.367,
        281.711,
        279.716,
        283.615,
        289.454,
        289.11,
        298.366,
        306.34,
        307.891,
        304.026,
        301.885, 308.461,
        309.498,
        291.242,
        293.776,
        302.233,
        389.16,
        ]
import matplotlib.pyplot as plt
if __name__ == "__main__":
    plt.figure(figsize=(8,10))
    data_new = list()
    # for dat in data:
    #     data_new.append(dat, dat)
    plt.grid(True)
    plt.plot(data, marker='o')
    plt.xlabel("checkpoints")
    plt.ylabel("FID score")
    plt.title("FID score evaluated at intermediate checkpoints")
    plt.show()