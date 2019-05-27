import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

# Set the LEDs
rows = 20
cols = 20
rows_use = rows // 2     # Actually is 4 * 4 (1/4 of the size)
cols_use = cols // 2
rows_rec = 40
cols_rec = 40
rows_rec_use = rows_rec // 2     # Actually is 5 * 5 (1/4 of the size)
cols_rec_use = cols_rec // 2
led_area = tf.ones(shape=[rows_use, cols_use])
rec = np.zeros(shape=[rows_use, cols_use, rows_rec_use, cols_rec_use], dtype=np.float32)

# The room size
length = 5
width = 5
height = 3

rec_input = np.loadtxt(open("G:\\tmp\\led_data\\1.csv", "rb"), delimiter=",", skiprows=0)
rec = np.reshape(rec_input, [rows_use, cols_use, cols_rec_use, rows_rec_use])   # The outcome shape is (20. 20, 20, 20)

def sum_from_4D_to_2D(input):
    shape = np.shape(input)
    result = np.zeros(shape=[shape[2], shape[3]])
    for i in range(shape[0]):
        for j in range(shape[1]):
            result += input[i][j]
    return result

def mut_num(input1, input2):
    shape = np.shape(input2)
    result = np.zeros(shape=[shape[0], shape[1], shape[2], shape[3]])
    for i in range(shape[0]):
        for j in range(shape[1]):
            result[i, j] = input1[i, j] * input2[i, j]
    return result

test_result = np.zeros(shape=[rows_use, cols_use, rows_rec_use, cols_rec_use])
test_result_1 = np.zeros(shape=[rows_use, cols_use, rows_rec_use, cols_rec_use])


position = 22
pos_x = position // cols_use
pos_y = position % rows_use
test_result[pos_x][pos_y] = 1.
# test_result[0][2] = test_result[7][0] = 1.

def plt_matrix(input):
    fig = plt.figure()
    ax = Axes3D(fig)
    Z = 1. * input[pos_x][pos_y]
    size = Z.shape
    Y = np.arange(0, size[0], 1)
    X = np.arange(0, size[1], 1)
    X, Y= np.meshgrid(X, Y)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    plt.show()

test_result_1[5][5] = 1.
z_test_1 = mut_num(rec, test_result_1)
z_2D_test_1 = sum_from_4D_to_2D(z_test_1)
MEAN_INIT = z_2D_test_1.mean()

def get_fitness(mean, var):     # The lower fitness, the better
    res = np.abs(MEAN_INIT - mean) + var
    res = 1 / res
    return res

z_test = mut_num(rec, test_result)
z_2D_test = sum_from_4D_to_2D(z_test) # shape of z_2D is (5, 5)
print(z_2D_test)
print('z_2D\n', z_2D_test)
var_test = z_2D_test.var()
mean_test = z_2D_test.mean()
print('The variance is ', round(var_test, 10))
print('The mean is ', round(mean_test, 10))
print('The fitness is ', get_fitness(mean_test, var_test))

plt_matrix(rec)
