import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import LSTM
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

# Define needed constants
N = 11      # Number of qubits
P = 10      # Number of gates per qubit
BS = 512    # Batch size

def DataLoad(N, P):

    """
    DataLoad returns the input to the neural network and the target values.
    The input is written via one-hot encoding.
    The target values are the expectation values in reverse order (i.e. the last component
    is the expectation value of the first qubit), because this is the default mode 
    in Qiskit.
    
    The parameters are the number of qubits N and the number of gates per qubit P.
    You can choose to get only the expectation value of the first qubit by using 
    first = True (default first = False)
    """

    # Load the input data
    inp = []
    circ = []
    with open(f'Dataset/input_N{N}_P{P}.dat') as file:
        for file in file:
            try:
                gates = file.split()[0]
                circ.append(float(gates))
            except:
                circ = np.array(circ)
                circ = circ.reshape(N, P)
                circ = circ.flatten('F')
                inp.append(circ)
                circ = []
                continue
    inp = np.array(inp, dtype=int)

    # Load the output data
    out = []
    for p in range(1, P+1):
        outputs = []
        listemp = []
        with open(f'Dataset/output_N{N}_P{p}.dat') as file:
            for line in file:
                try:
                    o = line.split()[0]
                    listemp.append(float(o))
                except:
                    outputs.append(listemp)
                    listemp = []
                    continue
        if p == 1:
            out = outputs.copy()
        else:
            for i in range(len(out)):
                out[i] = out[i] + outputs[i]
    out = np.array(out)

    return inp, out


def RnnModel(num_outputs):
    """
    CnnModel returns the convolutional neural network model
    
    num_outputs is needed to specify whether the network should predict the expectation
    values of all qubits or of the first qubit
    """
    rnn = Sequential()
    rnn.add(LSTM(units=(N*P), input_shape=(1, (N*P)), return_sequences=True, activation='relu'))                                              

    rnn.compile(
                loss='binary_crossentropy',   
                           optimizer='adam',
                          metrics=['mae', r2])
   
    return rnn

def r2(y_true, y_pred):
    """
    r2 returns the coefficient of determination R^2

    The parameters are the target values y_true and the values predicted by the neural 
    network y_pred
    """
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/SS_tot )

# Training Loop
if __name__ == '__main__':

    # Loading inputs and outputs of the neural network
    inputs, outputs = DataLoad(N, P)

    # Create a model and print its summary
    rnn = RnnModel(N)
    rnn.summary()
    # plot_model(rnn, to_file='Plots/rnn.png', show_shapes=True, show_layer_names=True, expand_nested=True, dpi=96)


    # Split in train and test set
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size = 0.2)

    # Delete the original inputs to save space
    del(inputs)
    del(outputs)

    # Expand the input data to match the shape expected by the RNN
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    # After 5 epochs (customizable) without any improvement (for the validation loss) 
    # the training stops and we get the best model weights    
    callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights = True)

    # Training the model 
    history = rnn.fit(X_train, y_train,
                        validation_split = 0.005,
                        epochs = 10,
                        verbose=1,
                        batch_size = BS,
                        callbacks = [callback])
    
    # Plot the training history of the model
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Plots/loss.png', dpi=300)

    # Save the weights
    rnn.save_weights(f'Weights/N{N}_P{P}__allqubits.h5')

    # Make a prediction on the test data
    y_pred_test = rnn.predict(X_test, batch_size=BS)
    y_pred_test = np.squeeze(y_pred_test)

    # Score the prediction
    print(r2_score(y_test, y_pred_test))
