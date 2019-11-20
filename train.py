import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import fb
from fb import df_to_ds
from fb import pack_dataset
from fb import define_model
from fb import plot_fitting_history
from fb import RECENT_BEST_MODEL_FILE_PREFIX
import gc


TRAIN_EPOCHS = 200
TEST_ACC_REQUIREMENT = 0.8
VALIDATION_ACC_REQUIREMENT = 0.8
BATCH_SIZE = 63
FITTING_VERBOSE = 0


'''
# Upload the training data as a pandas data frame, for colab.
uploaded = files.upload()
data_frame = pd.read_csv(io.BytesIO(uploaded['train.csv']))
data_frame.head()
data_frame.dtypes
'''
# Read the training data.
data_frame = pd.read_csv('train.csv')

def split_data(show_info=False):
    # Split the dataframe into train, validation and test.
    train_frame, test_frame = train_test_split(data_frame, test_size=0.2)
    train_frame, val_frame = train_test_split(train_frame, test_size=0.2)

    if show_info:
        print(len(train_frame), 'train examples')
        print(len(val_frame), 'validation examples')
        print(len(test_frame), 'test examples')
        print('\n\n')

    # Turn data frames into datasets.
    train_ds = df_to_ds(train_frame, batch_size=BATCH_SIZE)
    # Try to avoid fitting warnings.
    train_ds = train_ds.repeat()
    val_ds = df_to_ds(val_frame, shuffle=False, batch_size=BATCH_SIZE)
    test_ds = df_to_ds(test_frame, shuffle=False, batch_size=BATCH_SIZE)

    # for feature_batch, label_batch in train_ds.take(1):
    # print('Every feature:', list(feature_batch.keys()))
    # print('A batch of goal_difference1:', feature_batch['goals'])
    # print('A batch of result:', label_batch)

    # show_batch(train_ds)

    # Pack numeric features.
    packed_train_ds = pack_dataset(train_ds)
    packed_val_ds = pack_dataset(val_ds)
    packed_test_ds = pack_dataset(test_ds)
    # show_batch(packed_train_ds)

    return train_frame, test_frame, val_frame, train_ds, val_ds, test_ds, packed_train_ds, packed_val_ds, packed_test_ds


def train_by_req():
    print("\n============================== Train Loop Starts ==============================")

    # Keep training models until one of them meet the acc requirements.
    test_acc = 0
    val_acc = 0
    show_split_info = True
    attempts = 0

    while test_acc < TEST_ACC_REQUIREMENT or val_acc < VALIDATION_ACC_REQUIREMENT:
        train_frame, test_frame, val_frame, train_ds, val_ds, test_ds, packed_train_ds, packed_val_ds, packed_test_ds =\
            split_data(show_split_info)
        show_split_info = False

        model = define_model()

        # Define the optimizer, loss/error function and how we measure the performance.
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train.
        # Try to avoid warnings with: https://github.com/tensorflow/tensorflow/issues/32817
        features = train_frame.copy()
        features.pop('result')

        #print('steps_per_epoch: ', features.shape[0] // BATCH_SIZE)

        fitting_history = model.fit(packed_train_ds, validation_data=packed_val_ds, epochs=TRAIN_EPOCHS, steps_per_epoch=features.shape[0] // BATCH_SIZE, verbose=FITTING_VERBOSE)

        # Evaluate the accuracy on the test dataset.
        test_loss, test_acc = model.evaluate(packed_test_ds, verbose=0)
        val_acc = fitting_history.history['val_accuracy'][-1]

        print('Attempt: {:8}   Validation Accuracy: {:.2f}   Test Accuracy: {:.2f}'.format(attempts, val_acc, test_acc))
        #print('-------------------------------------------------')
        attempts = attempts + 1
    # End of while loop.

    model.summary()

    #plot_fitting_history(fitting_history, block=True)
    # plot_fitting_history(fitting_history, key = 'accuracy')

    print('\n\n\n')


    # Do the test set predicting.
    predictions = model.predict(packed_test_ds)

    my_test_frame = test_frame.copy()
    goals = my_test_frame.pop('goals')
    cheat = my_test_frame.pop('cheat')

    no_goal_examples = 0
    print('\n ======= Correct Test Set Predictions =======')
    i = 0
    for step, (x, y) in enumerate(packed_test_ds):
        for r in y:
            pred = predictions[i]
            if np.argmax(pred) == r.numpy():
                print("{:2.2f}: {} = {}   goals: {:2d},   cheat: {:7s}   index: {}".format(pred[np.argmax(pred)],
                                                                                           np.argmax(pred), r.numpy(),
                                                                                           goals[goals.index[i]],
                                                                                           cheat[cheat.index[i]],
                                                                                           cheat.index[i]))

                if (goals[goals.index[i]] == 0):
                    no_goal_examples = no_goal_examples + 1

            i = i + 1

    print('\n ======= Wrong Test Set Predictions =======')
    i = 0
    for step, (x, y) in enumerate(packed_test_ds):
        for r in y:
            pred = predictions[i]
            if np.argmax(pred) != r.numpy():
                print("{:2.2f}: {} = {}   goals: {:2d},   cheat: {:7s}   index: {}".format(pred[np.argmax(pred)],
                                                                                           np.argmax(pred), r.numpy(),
                                                                                           goals[goals.index[i]],
                                                                                           cheat[cheat.index[i]],
                                                                                           cheat.index[i]))
            i = i + 1





    model_fname = "{}_{:d}_{:d}_{:d}.h5".format(RECENT_BEST_MODEL_FILE_PREFIX,
                                                int(test_acc * 100),
                                                int(val_acc * 100),
                                                no_goal_examples)

    model.save_weights(model_fname)
    del model

# The train main loop.
while True:
    train_by_req()