import fb
import pandas as pd
from sklearn.model_selection import train_test_split


LAYER_DESC_OPTIONS = ((8, 8),)
EPOCHS_OPTIONS = (200, 300)
TOTAL_ATTEMPTS = 100
BATCH_SIZE = 57

result = []
for layer_desc in LAYER_DESC_OPTIONS:
    for epochs in EPOCHS_OPTIONS:
        average_accuracy, max_accuracy = fb.compare_trainning_step(layer_desc, epochs, TOTAL_ATTEMPTS, BATCH_SIZE)
        result.append('Model {}, epochs: {}       average accuracy = {:.2f}, max accuracy = {:.2f}'.format(
            layer_desc,
            epochs,
            average_accuracy,
            max_accuracy
        ))

print('\n!!!!!!! Comparing Result !!!!!!!')
for s in result:
    print(s)