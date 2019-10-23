import pandas as pd
import numpy as np
from fb import df_to_ds
from fb import pack_dataset
from fb import define_model
from fb import RECENT_BEST_MODEL_FILE

# Load the csv to predict.
predict_frame = pd.read_csv('predict.csv')
'''
uploaded = files.upload()
predict_frame = pd.read_csv(io.BytesIO(uploaded['predict.csv']))
'''

predict_ds = df_to_ds(predict_frame, shuffle=False)  # Notice: be sure shuffle being turned off.
packed_predict_ds = pack_dataset(predict_ds)

loaded_model = define_model()
loaded_model.predict(packed_predict_ds)  # Avoid error.
loaded_model.load_weights(RECENT_BEST_MODEL_FILE)

# Do the predicting.
predictions = loaded_model.predict(packed_predict_ds)

my_predict_frame = predict_frame.copy()
goals = my_predict_frame.pop('goals')
cheat = my_predict_frame.pop('cheat')

# Print the predictions.
i = 0
for step, (x, y) in enumerate(packed_predict_ds):
    for r in y:
        pred = predictions[i]
        print("{:2.2f}: {} = {}   goals: {:2d},   cheat: {:7s}   index: {}".format(
            pred[np.argmax(pred)],
            np.argmax(pred),
            r.numpy(),
            goals[goals.index[i]],
            cheat[cheat.index[i]],
            cheat.index[i]))
        i = i + 1
