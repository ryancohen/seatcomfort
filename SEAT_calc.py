"""
Calculates SEAT transmissability for a vibration profile, based on ISO 2631
frequency weighting.

Written R Cohen in Feb 2019
ME329
"""

import numpy as np
from matplotlib import pyplot as plt
import pdb
from scipy.fftpack import fft
import weighting

def weighted_power(x, t):
    N = len(t)  # number of data points
    T = (t[-1] - t[0])/len(t)  # average sample period
    Fs = 1 / T  # sample rate (Hz)
    f = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)
    xft = fft(x)
    xft = xft[0:np.int(N / 2)]
    psdx = (1/(Fs*N)) * np.abs(xft)**2
    psdx[1:-1] = 2*psdx[1:-1]
    w = np.zeros(len(f))
    for i in range(len(f)):
        w[i] = weighting.wb(f[i])
    return np.trapz(psdx*(w**2), f)

    # plt.plot(f, 2.0 / N * np.abs(a_rms))
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Accel (g)')
    # plt.title('Fourier Tranform')
    # plt.show()


if __name__ == "__main__":
    pairs = [('normal_seat_bottom_cushion_z.csv', 'normal_seat_mounting_points_z.csv'),
             ('30u_seat_bottom_cushion_z.csv', '30u_seat_mounting_points_z.csv'),
             ('30d_seat_bottom_cushion_z.csv', '30d_seat_mounting_points_z.csv'),
             ('30d_brace_seat_bottom_cushion_z.csv', '30d_brace_seat_mounting_points_z.csv'),
             ('30u_brace_seat_bottom_cushion_z.csv', '30u_brace_seat_mounting_points_z.csv')]
    for pair in pairs:
        print 'Analyzing data from ' + pair[0][:-4] + ' and ' + pair[1][:-4]
        s = np.genfromtxt(pair[0], delimiter=',', skip_header=2, unpack=True)
        f = np.genfromtxt(pair[1], delimiter=',', skip_header=2, unpack=True)
        t = s[0, :]

        f_switched = np.zeros(np.shape(f))  # Change ordering to correspond with cushion node ordering
        f_switched[1, :] = f[3, :]
        f_switched[2, :] = f[4, :]
        f_switched[3, :] = f[2, :]
        f_switched[4, :] = f[1, :]

        for i in range(1, 5):
            SEAT = np.sqrt(weighted_power(s[i, :], t)/weighted_power(f_switched[i, :], t))
            print 'SEAT% ' + str(i) + ' is: ' + str(SEAT*100) + '%'