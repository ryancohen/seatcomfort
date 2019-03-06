"""
Calculates VDV (Vibration Dose Value) for a vibration profile, based on ISO 2631
frequency weighting.

Written R Cohen in Feb 2019
ME329
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy.fftpack import fft

def calc_VDV(vibs, t):
    N = len(t)  # number of data points
    T = (t[1] - t[0])  # sample period
    T = 0.0001 #TODO UNHARDCODE THIS
    Fs = 1 / T  # sample rate (Hz)
    f = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)
    a_rms = fft(x)
    plt.plot(f, 2.0 / N * np.abs(a_rms[0:np.int(N / 2)]))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Accel (g)')
    plt.title('Fourier Tranform')
    plt.show()
    VDV = 0.0
    for i in range(N/2):
        VDV += np.abs(a_rms[i])*wb(f[i])*2.0 / N
    return VDV*(t[-1] - t[0])**0.25

def wb(f):
    if f > 1000:
        return 0.001
    if f > 100:
        return 0.01
    if f > 10:
        return 0.1
    return 0.5


def generate_rand_profile(t_max):
    t = np.arange(0, t_max, 10**(-3))
    vibs = np.random.randn(len(t))
    pdb.set_trace()
    return vibs, t


if __name__ == "__main__":
    #vibs, t = generate_rand_profile(t_max = 60)
    file_path = 'commute.csv'
    t, x = np.genfromtxt(file_path, delimiter=',', unpack=True)
    plt.plot(t, x)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Accel (g)')
    plt.title('Vehicle Vibration Profile')
    plt.show()

    print t.shape

    VDV = calc_VDV(x, t)
    print VDV*9.81
