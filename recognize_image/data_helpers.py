
def unpickle():
    import pickle
    file = 'G:\\Python35\\workspace\\tmy\\Deep_Learning\\recognize_image\\cifar-10-batches-py\\data_batch_2'
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict