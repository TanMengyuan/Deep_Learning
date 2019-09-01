#    (0, 0)--→  -----------------  (0, 5)
#    rows       |   |   |   |   |
#               |---------------|
#               |   |   |   |   |    width = 5 m
#               |---------------|
#               |   |   |   |   |
#               |---------------|
#               |   |   |   |   |
#     (5, 0)    -----------------    16 LEDs in the room
#                   length = 5 m
#               cols


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

is_create_data = False

def cal_incidence(local):
    # return a 2D array for the receiver incidence
    # local[0] is the x, local[1] is the y
    tmp = np.zeros(shape=[rows_rec, cols_rec])
    for x_pos in range(rows_rec):
        for y_pos in range(cols_rec):
            x = np.abs(local[0] - (x_pos + 0.5) * (width / cols_rec))
            y = np.abs(local[1] - (y_pos + 0.5) * (length / rows_rec))
            d = np.sqrt(np.square(x) + np.square(y) + np.square(height))
            tmp[x_pos][y_pos] = round((np.square(1.6) * 0.73 / d ** 4), 4)  # This formula might be change
    result = np.zeros(shape=[rows_rec_use, cols_rec_use])
    for i in range(rows_rec_use):
        for j in range(cols_rec_use):
            result[i][j] = tmp[i][j] + tmp[j][rows_rec - i - 1] + tmp[rows_rec - i - 1][cols_rec - j - 1] + tmp[cols_rec - j - 1][i]
    return result

def plt_matrix(input):
    fig = plt.figure()
    ax = Axes3D(fig)
    Z = 1. * input[1][1]
    size = Z.shape
    Y = np.arange(0, size[0], 1)
    X = np.arange(0, size[1], 1)
    X, Y= np.meshgrid(X, Y)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    plt.show()

if is_create_data:
    for i in range(rows_use):
        for j in range(cols_use):
            loc = [(width / rows) * (i + 0.5), (length / cols) * (j + 0.5)]
            rec[i][j] = cal_incidence(loc)
    rec_outcome = np.reshape(rec, (rows_use * cols_use, -1))     # The shape of rec_outcome is (20, 20, -1)
    np.savetxt('G:\\tmp\\led_data\\1.csv', rec_outcome, delimiter = ',')
else:
    rec_input = np.loadtxt(open("G:\\tmp\\led_data\\1.csv", "rb"), delimiter=",", skiprows=0)
    rec = np.reshape(rec_input, [rows_use, cols_use, cols_rec_use, rows_rec_use])   # The outcome shape is (20. 20, 20, 20)

# print(rec[1][1])
# plt_matrix(rec)
#
# Z = 1.0 * rec[3][3] + 1.0 * rec[3][16] + 1.0 * rec[16][3] + 1.0 * rec[16][16]
# print(np.mean(Z))

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

# test_result[1][1] = 1.
# z_test = mut_num(rec, test_result)
# z_2D_test = sum_from_4D_to_2D(z_test) # shape of z_2D is (5, 5)
# print(z_2D_test)
# print('z_2D\n', z_2D_test)
# var_test = z_2D_test.var()
# mean_test = z_2D_test.mean()
# print('The variance is ', round(var_test, 10))
# print('The mean is ', round(mean_test, 10))

########################################################
######-----------Genetic Algorithm----------------######
########################################################

# test_result[rows_use // 2][cols_use // 2] = 1.
test_result[4][4] = 1.
z_test = mut_num(rec, test_result)
z_2D_test = sum_from_4D_to_2D(z_test)
mean_use = z_2D_test.mean()

DNA_SIZE = rows_use * cols_use
POP_SIZE = 100
CROSS_RATE = 0.8
MUTATION_RATE = 0.003
N_GENERATIONS = 1000
MEAN_INIT = mean_use

print('mean init is ', MEAN_INIT)

def F(x):       # x is an array of LED, and it will return two variable
    x = np.reshape(x, newshape=[rows_use, cols_use])
    tmp = np.zeros(shape=[rows_use, cols_use, rows_rec_use, cols_rec_use])
    for i in range(rows_use):
        for j in range(cols_use):
            tmp[i][j] = x[i][j]
    z = mut_num(rec, tmp)
    z_2D = sum_from_4D_to_2D(z)
    mean = z_2D.mean()
    var = z_2D.var()
    return mean, var

def get_fitness(mean, var):     # The lower fitness, the better
    res = np.abs(MEAN_INIT - mean) + np.multiply(var, 1e4)
    res = 1 / res
    return res

def select(pop, fitness):  # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness / fitness.sum())
    return pop[idx]

def crossover(parent, pop):  # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)  # select another individual from pop
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)  # choose crossover points
        parent[cross_points] = pop[i_, cross_points]  # mating and produce one child
    return parent

def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child

def myappend(list, value):
    list_1 = list.copy()
    list_1.append(value)
    return list_1

pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))  # initialize the pop DNA
print(pop[:10])

for num in range(N_GENERATIONS):
    F_mean = F_var = []
    for i in range(POP_SIZE):
        F_tmp = F(pop[i])
        F_mean = myappend(F_mean, F_tmp[0])
        F_var = myappend(F_var, F_tmp[1])
    fitness = get_fitness(F_mean, F_var)
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parant in pop:
        child = crossover(parant, pop_copy)
        child = mutate(child)
        parant[:] = child
    if num % 50 == 0:
        print('At the', num, 'generation, the distribution is:')
        for n in range(10):
            idx = np.where(pop[n] == 1)
            print('The prefect position is', idx)


########################################################
######-------------Start training-----------------######
########################################################


# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)
#
# def sum_up_gap(input):
#     mean = np.mean(input)
#     var = np.var(input)
#     # rel_rec = np.zeros(shape=[1, np.shape(input)[1]])
#     # print(x, y)
#     result = np.square(mean - var)
#     return result
#
# init_data = tf.placeholder(dtype=tf.float32, shape=[rows, cols], name='LED_position')
# # ideal_dis = tf.placeholder(dtype=tf.float32, shape=[rows_rec, cols_rec], name='ideal_distribution')
# keep_prob = tf.placeholder(tf.float32)
#
# data_input = tf.reshape(init_data, shape=[1, rows * cols])
# W_layer1 = weight_variable([1, rows * cols])
# b_layer1 = bias_variable([1, rows * cols])
# h_layer1 = tf.add(tf.multiply(data_input, W_layer1), b_layer1)     # layer1 output shape is (1, 1024)
#
# W_layer2 = weight_variable([rows * cols, 1024])
# b_layer2 = bias_variable([1, 1024])
# h_layer2 = tf.nn.relu(tf.matmul(h_layer1, W_layer2) + b_layer2)
#
# W_layer3 = weight_variable([1024, rows * cols])
# b_layer3 = bias_variable([1, rows * cols])
# pred = tf.nn.softmax(tf.matmul(h_layer2, W_layer3) + b_layer3)      # pred's shape is (1, 16)
#
# # opt_pos = tf.reshape(pred, shape=[rows, cols])
# gap = sum_up_gap(pred)
# train_step = tf.train.AdamOptimizer(1e-4).minimize(gap)
#
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
#
# for i in range(1000):
#     if i % 50 == 0:
#         pass
#         # train_accuracy = accuracy.eval(feed_dict={
#         #     x: batch[0], y_: batch[1], keep_prob: 1.0})
#         # print("step %d, training accuracy %g" % (i, train_accuracy))
#     train_step.run(feed_dict={init_data: led_area, keep_prob: 1})