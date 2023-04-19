import elements as e
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


class RForest:
    def __init__(self,Tree_model, depth, min_samples, trees_count):
        self.Tree_model = Tree_model
        self.trees = []
        self.depth = depth
        self.min_samples = min_samples
        self.trees_count = trees_count

    def fit(self, samples, results):
        for i in range(self.trees_count):
            tree = self.Tree_model(self.depth, self.min_samples)
            tree.fit(samples, results)
            self.trees.append(tree)

    def predict(self, x):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(x))
        return e.most_common(predictions)

    def mass_predict(self, x):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.mass_predict(x))
        return e.most_common(predictions)

    def save_forest(self, filename):
        pickle.dump(self, open(filename, "wb"))

    def load_forest(self, filename):
        return pickle.load(open(filename, "rb"))
