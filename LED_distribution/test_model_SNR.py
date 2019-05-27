import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import scipy

hparams_justify = 1
# hparams_justify = 4 / 3

plt.figure(figsize=(12, 6))  # set the figure size
ROOM_SIZE = np.array([10, 10])
DNA_SIZE = ROOM_SIZE[0] * ROOM_SIZE[1]            # DNA length
dimX, dimY, dimZ, REC_HEIGHT = 5, 5, 3, 0.85
ngx, ngy = dimX * 10, dimY * 10
ht, hr = dimZ, REC_HEIGHT
htr = ht - hr
x = np.linspace(0 + dimX / (2 * ngx), dimX - dimX / (2 * ngx), ngx)
y = np.linspace(0 + dimY / (2 * ngy), dimY - dimY / (2 * ngy), ngy)
xr, yr = np.meshgrid(x, y)
xt = np.linspace(0 + 0.25, 5 - 0.25, 10)
yt = np.linspace(0 + 0.25, 5 - 0.25, 10)

c = 3e8 # m/s
nLed = 60
# nLed = 60
Pt = 0.02 # W
Pt *= nLed * nLed
gamma = 0.53 # A/W
# data_rate = 2000 * 1e6 # b/s
data_rate = 10 * 1e6 # b/s
T = 1 / data_rate # s
T_half = T / 2

noise_value_data, t_value_data, Hn_value_data = np.array([]), np.array([]), np.array([])
ambient_noise_value_data, n_shot_noise_value_data, n_thermal_noise_value_data = np.array([]), np.array([]), np.array([])
E_value_data = np.array([])

for i in range(ROOM_SIZE[0]):
    for j in range(ROOM_SIZE[1]):
        E_value_data = np.append(E_value_data,
                                     np.load(r'E_value_data_onetime_reflection/E_value_%s.npy' % (str(i) + str(j))))
        noise_value_data = np.append(noise_value_data,
                                 np.load(r'noise_value_data/noise_value_%s.npy' % (str(i) + str(j))))
        t_value_data = np.append(t_value_data,
                                     np.load(r't_value_data/t_value_%s.npy' % (str(i) + str(j))))
        Hn_value_data = np.append(Hn_value_data,
                                 np.load(r'Hn_value_data/Hn_value_%s.npy' % (str(i) + str(j))))
        n_shot_noise_value_data = np.append(n_shot_noise_value_data,
                                     np.load(r'n_shot_value_data/n_shot_value_%s.npy' % (str(i) + str(j))))
        n_thermal_noise_value_data = np.append(n_thermal_noise_value_data,
                                     np.load(r'n_thermal_value_data/n_thermal_value_%s.npy' % (str(i) + str(j))))
        if os.path.isfile(r'ambient_noise_value_data/ambient_noise_value_%s.npy' % (str(i) + str(j))):
            ambient_noise_value_data = np.append(ambient_noise_value_data,
                                         np.load(r'ambient_noise_value_data/ambient_noise_value_%s.npy' % (str(i) + str(j))))
        else:
            ambient_noise_value_data = np.append(ambient_noise_value_data, np.zeros((50, 50)))


noise_value_data = noise_value_data.reshape(ROOM_SIZE[0], ROOM_SIZE[1], 50, 50)
ambient_noise_value_data = ambient_noise_value_data.reshape(ROOM_SIZE[0], ROOM_SIZE[1], 50, 50)
t_value_data = t_value_data.reshape(ROOM_SIZE[0], ROOM_SIZE[1], 50, 50)
Hn_value_data = Hn_value_data.reshape(ROOM_SIZE[0], ROOM_SIZE[1], 50, 50)
n_shot_noise_value_data = n_shot_noise_value_data.reshape(ROOM_SIZE[0], ROOM_SIZE[1], 50, 50)
n_thermal_noise_value_data = n_thermal_noise_value_data.reshape(ROOM_SIZE[0], ROOM_SIZE[1], 50, 50)
E_value_data = E_value_data.reshape(ROOM_SIZE[0], ROOM_SIZE[1], 50, 50)


def plotting(DNA, id_num):
    room_id = str(id_num).zfill(3)
    room = np.load('room_data/%s.npy' % room_id)
    # print(room)
    room = np.ones((10, 10))
    room_area = len(np.where(room == 1)[0])
    repeat_arr = np.ones(10, dtype=np.int) * 5
    room_mut = np.repeat(room, repeat_arr, axis=0)
    room_mut = np.repeat(room_mut, repeat_arr, axis=1)
    x, y = np.array([]), np.array([])
    room_xx, room_yy = np.where(room == 0)[0] / 2 + 0.25, np.where(room == 0)[1] / 2 + 0.25

    DNA = DNA.reshape(-1, ROOM_SIZE[0], ROOM_SIZE[1])[0]
    xt, yt = [], []
    S, N, E, ambient_noise, t_min = np.zeros((ngx, ngy)), np.zeros((ngx, ngy)), np.zeros((ngx, ngy)), \
                                    np.zeros((ngx, ngy)), np.zeros((ngx, ngy))
    # test = np.zeros((ngx, ngy))
    indexes = np.where(DNA == 1)
    led = len(indexes[0])
    win_xx, win_yy = [], []

    # for i in range(0, 10):
    #     N += ambient_noise_value_data[i][0]
    #     win_xx.append(i / 2 + 0.25)
    #     win_yy.append(-0.1)

    for j in range(led):
        xt.append(indexes[0][j])
        yt.append(indexes[1][j])
        x = np.append(x, indexes[0][j] / 2 + 0.25)
        y = np.append(y, indexes[1][j] / 2 + 0.25)

    for l in range(len(xt)):
        t_min = np.minimum(t_min, t_value_data[xt[l]][yt[l]])

    print(xt)
    print(yt)
    for k in range(len(xt)):
        E += E_value_data[xt[k]][yt[k]]

    E *= nLed * nLed * room_mut
    min_E = np.min(E[E > 0])
    amp = 300 / min_E
    # hparams_justify = amp
    # E *= amp

    for k in range(len(xt)):
        # ts = t_value_data[xt[k]][yt[k]] - t_min
        Hn = Hn_value_data[xt[k]][yt[k]]
        # Prs, ISI = np.zeros((50, 50)), np.zeros((50, 50))
        # Prs[ts <= T_half] = Pt * Hn[ts <= T_half]
        # ISI[ts > T_half] = Pt * Hn[ts > T_half]
        Prs = Pt * Hn * hparams_justify
        S += (gamma ** 2) * (Prs ** 2)

        # N += noise_value_data[xt[k]][yt[k]] + (gamma ** 2) * ((Pt * Hn) ** 2)
        # N += noise_value_data[xt[k]][yt[k]] + (gamma ** 2) * (ISI ** 2)
        # N += noise_value_data[xt[k]][yt[k]]
        # N += n_thermal_noise_value_data[xt[k]][yt[k]] + n_shot_noise_value_data[xt[k]][yt[k]] * hparams_justify
        N += 0.1 * (gamma ** 2) * (Prs ** 2)


    SNR = 10 * np.log10(S / N) * room_mut
    print('mean SNR', np.mean(SNR))
    print('min SNR ', np.min(SNR[SNR > 0]))
    print('max SNR ', np.max(SNR), '\n')

    ratio = len(SNR[SNR > 13.6]) / (room_area * 25)
    print('Effect Area : {0}'.format(ratio))

    print(np.min(E[E > 0]))

    # plt.subplot(121)
    # # plt.contourf(xr, yr, E.T, alpha=.75)
    # # C = plt.contour(xr, yr, E.T, colors='black', linewidths=1)
    # plt.contourf(xr, yr, SNR.T, alpha=.75)
    # C = plt.contour(xr, yr, SNR.T, colors='black', linewidths=1)
    # plt.clabel(C, fmt='%.1f', inline=True, fontsize=10)


    plt.subplot(121)
    levels = np.hstack((np.linspace(np.min(SNR[SNR != 0]), 13.6 - (np.max(SNR) - 13.6) / 3, 3),
                        np.linspace(13.6 + (np.max(SNR) - 13.6) / 4, np.max(SNR), 4))) \
        if np.max(SNR) > 13.6 else np.linspace(0, np.max(SNR), 8)
    plt.contourf(xr, yr, SNR.T, levels=levels, alpha=.75)
    C = plt.contour(xr, yr, SNR.T, levels=levels, colors='black', linewidths=1)
    C_ = plt.contour(xr, yr, SNR.T, levels=[np.min(SNR[SNR != 0]), 13.6], colors='black', linewidths=3)
    plt.clabel(C, fmt='%.1f', inline=True, fontsize=10, manual=True)
    plt.clabel(C_, fmt='%.1f', inline=True, fontsize=10, manual=True)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('SNR (dB) Effect Area: {0} %'.format(round(round(ratio, 4) * 100, 2)))

    plt.subplot(122)
    plt.scatter(x, y)
    plt.scatter(room_xx, room_yy, s=[1200], marker='s', c='gray')
    plt.scatter(win_xx, win_yy, s=[1200], marker='s', c='blue', alpha=0.6)
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    # plt.title('Generations : %d ' % gen)
    plt.title('room model')

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(xr, yr, SNR.T, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    # ax.plot_surface(xr, yr, S.T, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    # ax.plot_surface(xr, yr, N.T, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    # ax.set_xlabel('X (m)')
    # ax.set_ylabel('Y (m)')
    # ax.set_zlabel('SNR (dB)')
    # ax.zaxis.get_major_formatter().set_powerlimits((0, 1))
    # ax.set_zlabel('noise')
    # ax.set_zlim(10, 26)
    plt.show()


dna = np.zeros((1, 100))
li = []
d = 2
nd = 9 - d
# dna[0][d * 11] = dna[0][nd * 10 + d] = dna[0][d * 10 + nd] = dna[0][nd * 11] = 1
# li = [27, 42, 75]
# li = [17, 42, 57, 82]
if li:
    for each in li:
        dna[0][each] = 1
# dna[0][11] = dna[0][71] = dna[0][75] = dna[0][27] = 1
# dna[0][22] = dna[0][23] = dna[0][32] = dna[0][33] = 1
# dna[0][88] = 1
# id_num = np.load('room_result_SNR/log.npy')
id_num = 13
plotting(dna, id_num)