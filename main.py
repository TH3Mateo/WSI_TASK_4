import numpy as np
import time
import ID3
from random_forest import RForest
import matplotlib.pyplot as plt
import elements as e
import pandas as pd
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report
s = time.time()

data_x, data_y,selected_config = e.setup("cars")
tree_count  = 20
depth = 6

data_x = e.translate(data_x,selected_config)
train_x, train_y, test_x, test_y = e.datasplitter(data_x,data_y,20)

train_x, train_y = e.datashuffler(train_x,train_y)




# classifier = ID3.ID3_tree(5,2)
# classifier.fit(train_x,train_y)
# classifier.save_tree("storage.pkl")
#
# forest = RForest(ID3.ID3_tree,depth,2,tree_count)
# forest.fit(train_x,train_y)
# forest.save_forest("storage.pkl")



new_classifier = pickle.load(open("storage.pkl","rb"))
classifier_output = new_classifier.mass_predict(test_x)


effect = e.encode(classifier_output,selected_config['encodes']['decision'])
expected = e.encode(np.array(test_y),selected_config['encodes']['decision'])
diff = np.array(expected)-np.array(effect)
# print(diff)
b_diff = np.array(np.array(diff,dtype=bool),dtype = int)

print("TIME ELAPSED: " + str(time.time()-s))


cm = confusion_matrix(expected, effect)
cm_df = pd.DataFrame(cm, selected_config['encodes']['decision'], columns=selected_config['encodes']['decision'])

# Plot confusion matrix
sns.heatmap(cm_df, annot=True, cmap='Greens',fmt = 'g')
# sns.heatmap(classification_report(expected, effect, target_names=selected_config['encodes']['decision']),annot=True, cmap='Reds')
#
plt.xlabel("predictions",fontsize = 16)
plt.ylabel("expectations",fontsize = 16)
title = "DEPTH: " + str(depth) + "\n" + "TREE COUNT: " + str(tree_count)
plt.title(title, fontsize =20)

print(classification_report(expected, effect, target_names=selected_config['encodes']['decision']))


plt.show()


# plt.show()

