import fb
import pandas as pd
from sklearn.model_selection import train_test_split


LAYER_DESC_OPTIONS = ((7, 5), (7, 7))
EPOCHS_OPTIONS = (200, 300, 400, 500, 800, 1000)
#EPOCHS_OPTIONS = (500, 800, 1000)
TOTAL_ATTEMPTS = 10
BATCH_SIZE = 51





for layer_desc in LAYER_DESC_OPTIONS:
    #model = fb.define_model_ex(layer_desc)
    model = fb.define_model()
    for epochs in EPOCHS_OPTIONS:
        average_accuracy = fb.train_step(model, epochs, TOTAL_ATTEMPTS, BATCH_SIZE)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Model {}, epochs: {}       average accuracy = {:.2f}\n\n'.format(layer_desc, epochs, average_accuracy))