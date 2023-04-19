import pandas as pd
import numpy as np
import json
import os
import ID3
import elements as e
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle



data_x, data_y,selected_config = e.setup("cars")
data_x = e.translate(data_x,selected_config)
train_x, train_y, test_x, test_y = e.datasplitter(data_x,data_y,20)

train_x, train_y = e.datashuffler(train_x,train_y)




classifier = ID3.ID3_tree(4,2)
classifier.fit(train_x,train_y)
classifier.save_tree("storage.pkl")



new_classifier = pickle.load(open("storage.pkl","rb"))
effect = pd.DataFrame(new_classifier.mass_predict(test_x))



os.system("cls")
print(np.transpose(np.array(effect)))
print(np.array(test_y))
converter  = ["none","unacc","acc","good","vgood"]



