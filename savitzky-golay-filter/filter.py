from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter



def noisy_signal():
    np.random.seed(0)
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, x.size)

    plt.plot(x, y, label="Noisy Signal")
    plt.grid(lw=2, ls=':')
    plt.xlabel("TimeStep")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.show()

def smoothing():
    np.random.seed(0)
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, x.size)

    window_size = 11
    degree = 3
    y_smooth = savgol_filter(y, window_size, degree)

    plt.plot(x, y, label="Noisy Signal")
    plt.plot(x, y_smooth, label="Smoothed Signal")
    plt.grid(lw=2, ls=':')
    plt.xlabel("TimeStep")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    noisy_signal()
    smoothing()
