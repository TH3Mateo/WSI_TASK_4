import pandas as pd
import numpy as np
import json


class Node:
    def __init__(self, feature = None, condition_val  = None,right = None, left = None, inf_gain = None, value = None):
        self.feature = feature
        self.condition_val = condition_val
        self.right = right
        self.left = left
        self.inf_gain = inf_gain
        self.value = value


def setup(dataset_name):

    selected_data = json.load(open('options.json'))[dataset_name]
    columns = selected_data['columns']
    full_data = pd.read_csv(selected_data['filename'], names=columns)
    data_x = full_data.drop(columns[-1], axis=1)
    data_y = full_data[columns[-1]]
    return data_x, data_y, selected_data


def information_gain(y,left,right):

    return gini(y) - (len(left)/len(y))*gini(left) - (len(right)/len(y))*gini(right)


def gini(y):
    ''' function to compute gini index '''

    class_labels = np.unique(y)
    g = 0
    for cls in class_labels:
        p_cls = len([s for s in y if s==cls]) / len(y)
        g += p_cls ** 2
        # print("gini: ",1-g)
    return 1 - g

def splitter(x,y,samples_count, feature_count):

    df = pd.concat([x,y],axis = 1)

    max_gain = -float('inf')
    names = x.columns.values
    # print(feature_count)

    for i in range(feature_count):

        thresholds = np.unique(x.iloc[:,i])


        for threshold in thresholds:
            # print(threshold)
            # print(x.iloc[1][i])
            left = df.loc[x[names[i]]<=threshold]
            # print(x[names[i]])
            right = df.loc[x[names[i]] > threshold]


            gain = information_gain(y,left[df.columns.values[-1]],right[df.columns.values[-1]])
            # print(gain)
            if gain > max_gain:
                max_gain = gain

                best_effect = {"feature": i, "condition_value": threshold,
                   "left_x": left.drop(["decision"], axis='columns'),
                   "right_x": right.drop(["decision"], axis='columns'),
                   "left_y": left[df.columns.values[-1]], "right_y": right[df.columns.values[-1]], "gain": max_gain}
    return best_effect

def most_common(y):
    return y.value_counts().idxmax()


def datashuffler(x, y):
    combo = pd.concat([x, y], axis=1)
    combo = combo.sample(frac=1).reset_index(drop=True)
    ox = combo.drop(combo.columns[-1], axis=1)
    oy = combo[combo.columns[-1]]
    return ox, oy




def datasplitter(x,y,test_percent):
    test_size = int(len(x)*(test_percent/100))
    x,y = datashuffler(x,y)

    test_x = x[:test_size]
    test_y = y[:test_size]
    train_x = x[test_size:]
    train_y = y[test_size:]
    return train_x, train_y, test_x, test_y


def encode(column,keys):
    return [keys.get(record) for record in column]

def translate(x,config):
    # print(type(x))
    out= []
    # names = x.columns.values()


    for n in range(len(x.iloc[0])):
        # print(x.columns.values)
        # print(n)
        current = list(x.columns.values)[n]
        # print(current)

        if current in config['encodes']:
            # out[n] =encode(x.iloc[:,n],config['encodes'][current])
            out.append(encode(x.iloc[:,n],config['encodes'][current]))
        else:
            out.append(x.iloc[:,n])

    return pd.DataFrame(np.transpose(out), columns=x.columns.values)

