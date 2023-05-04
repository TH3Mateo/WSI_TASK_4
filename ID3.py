import numpy as np
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
import elements as e
from elements import Node


class ID3_tree:
    def __init__(self, depth, min_samples):

        self.root = None
        self.depth = depth
        self.min_samples = min_samples

    def build(self, samples,results,current_depth=0):
        # print("Building tree: depth ",current_depth)
        sample_count, features_count = np.shape(samples)
        if sample_count>= self.min_samples and current_depth <= self.depth:
            # print("condition passed")

            best = e.splitter(samples,results,sample_count,features_count)
            # print(best["gain"])
            if best["gain"] >0:
                # print("gain is greater than 0")
                left_branch = self.build(best["left_x"],best["left_y"],current_depth+1)
                # print("left branch built")
                right_branch = self.build(best["right_x"],best["right_y"],current_depth+1)
                # print("returning Node")
                return Node(best["feature"],best["condition_value"],right_branch,left_branch,best["gain"])

        # print("returning leaf")
        return Node(value = e.most_common(results))

    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        print(self.root)
        if not tree:
            tree = self.root
            print(tree)

        if tree.value :
            print(tree.value)

        else:
            print("X_" + str(tree.feature), "<=", tree.condition_val, "?", tree.inf_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

        def fit(self, samples, results):
            self.root = self.build(samples, results)


    def predict(self, x, tree=None):
        ''' function to make predictions '''

        if tree.value != None:
            return tree.value
        feature_val = x[tree.feature]
        if feature_val <= tree.condition_val:
            return self.predict(x, tree.left)
        else:
            return self.predict(x, tree.right)



    # def predict(self,x,tree=None):
    #

    def fit(self, samples, results):
        self.root = self.build(samples, results)
        # print(self.root)

    def mass_predict(self, x):
        return np.array([self.predict(x.iloc[i],self.root) for i in range(len(x))])

    def save_tree(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f,pickle.HIGHEST_PROTOCOL)