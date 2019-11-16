import pandas as pd
import numpy as np
from fb import df_to_ds
from fb import pack_dataset
from fb import define_model

PREDICTING_MODEL_FNAME = 'recent_best_weights_80_81_7.h5'
FRESH_DATA_FILE = 'fresh_data.csv'
FEATURES_IN_PREDICTION = ['round', 'home', 'away', 'cheat',
                          'advantage0', 'advantage1', 'advantage2', 'advantage3', 'advantage4',
                          'defense_home', 'defense_away', 'goals']
PREDICT_BATCH_SIZE = 32

# Load the fresh data for predicting.
fresh_data_frame = pd.read_csv(FRESH_DATA_FILE)
fresh_ds = df_to_ds(fresh_data_frame, shuffle=False, batch_size=PREDICT_BATCH_SIZE)  # Notice: be sure shuffle being turned off.
packed_fresh_ds = pack_dataset(fresh_ds)

# Prepare the model.
loaded_model = define_model()
loaded_model.predict(packed_fresh_ds)  # Avoid error.
loaded_model.load_weights(PREDICTING_MODEL_FNAME)

# Do the predicting.
predictions = loaded_model.predict(packed_fresh_ds)

# Show the prediction.
example_count = 0
correct_count = 0
# for each ds batch which is a dictionary mapping each feature batch to its corresponding label array.
for feature_batch_dict, label_array in fresh_ds:
    # for each example in batch.
    for example_index in range(PREDICT_BATCH_SIZE):
        print("\n\n- Prediction {}: ".format(example_count + 1), end='       ')
        real_result = label_array[example_index]
        predicted_result = np.argmax(predictions[example_count])
        if (real_result == predicted_result):
            correct_count = correct_count + 1

            print("{}({:2.2f})              ".format(
                predicted_result,
                predictions[example_count][predicted_result]),
                end='              ')
        else:
            print("{}({:2.2f}) -> {} Wrong!!!".format(
                predicted_result,
                predictions[example_count][predicted_result],
                real_result),
                end='              ')

        for feature in FEATURES_IN_PREDICTION:
            print(feature, '=', feature_batch_dict[feature][example_index].numpy(), end=' | ')
        example_count = example_count + 1
        if example_count >= len(predictions):
            break
    if example_count >= len(predictions):
        break

print("\n\n-------------- Prediction Accuracy: {:2.2f}   with: {}--------------".format(correct_count / example_count, PREDICTING_MODEL_FNAME))