import numpy as np

def file_len(filename):
    with open(filename) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def cars_load(filename,samples):
    out = []
    counter = 0
    with open(filename, 'r') as f:

        for line in f:
            if counter < samples:

                out.append(line.split(','))
                counter += 1
            else:
                break
    return out

x= cars_load('test.data',file_len('test.data'))
print(np.array(x)[-1])