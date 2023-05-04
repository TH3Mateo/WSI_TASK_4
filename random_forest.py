import sys
from threading import Thread
import numpy as np
import elements as e
import random
from statistics import mode
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle





class RForest:
    def __init__(self,Tree_model, depth, min_samples, trees_count,dropped_cols=None):
        self.Tree_model = Tree_model
        self.trees = []* trees_count
        self.depth = depth
        self.min_samples = min_samples
        self.trees_count = trees_count
        self.dropped_cols = dropped_cols


    def fit(self, samples, results):
        for i in range(self.trees_count):
            tree = self.Tree_model(self.depth, self.min_samples)

            samples,results = e.datashuffler(samples,results)
            # print(samples.columns)
            for j in range(self.dropped_cols if self.dropped_cols else random.randint(1,len(samples.columns))):
                samples.drop(random.choice(samples.columns),axis=1)
            tree.fit(samples, results)
            self.trees.append(tree)




    def unify(self,predictions):
        def oftenest(list):
            values,counts = np.unique(list, return_counts=True)
            return values[counts.argmax()]
        predictions = np.transpose(np.array(predictions))
        out = []
        # print(predictions[0])
        for row in predictions:
            # print(row)
            out.append(oftenest(row))
        return out
    def mass_predict(self, x):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.mass_predict(x))
        return self.unify(predictions)

    def save_forest(self, filename):
        pickle.dump(self, open(filename, "wb"))

