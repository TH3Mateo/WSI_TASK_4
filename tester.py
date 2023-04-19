def predict(self, X):
    ''' function to predict new dataset '''

    preditions = [self.make_prediction(x, self.root) for x in X]
    return preditions


def make_prediction(self, x, tree):
    ''' function to predict a single data point '''

    if tree.value != None: return tree.value
    feature_val = x[tree.feature_index]
    if feature_val <= tree.threshold:
        return self.make_prediction(x, tree.left)
    else:
        return self.make_prediction(x, tree.right)


Train - Test
split