import numpy as np

id_num = np.load('log.npy')
room_id = str(id_num).zfill(3)
file = room_id + '_out.npy'
new_file = './revised_data/' + file

ori_data = np.load(file)
print('id = ', id_num)
print(np.where(ori_data == 1)[1])

dna = np.zeros((1, 100))
li = [25, 52, 67]
for each in li:
    dna[0][each] = 1

# np.save('log.npy', id_num + 1)
# np.save(new_file, dna)
print('Saved.')