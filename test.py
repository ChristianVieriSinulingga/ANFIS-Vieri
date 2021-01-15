import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from anfis1 import ANFIS


# # Generate dataset
D = 4  # number of regressors
ts = np.loadtxt("Data-V2.csv", delimiter=",", skiprows=1,)
data = ts[:, 0:4]
lbls = ts[:, 0]

trnData = data
trnLbls = lbls
chkData = data[lbls.size - round(lbls.size * 0.15):, :]
chkLbls = lbls[lbls.size - round(lbls.size * 0.15):]

# ANFIS params and Tensorflow graph initialization
m = 4  # number of rules
alpha = 0.02  # learning rate

fis = ANFIS(n_inputs=D, n_rules=m, learning_rate=alpha)

# Training
num_epochs = 20000
print(len(trnData), len(trnLbls), len(chkData), len(chkLbls))
# Initialize session to make computations on the Tensorflow graph
with tf.Session() as sess:
    # Initialize model parameters
    sess.run(fis.init_variables)
    trn_costs = []
    val_costs = []
    time_start = time.time()
    for epoch in range(num_epochs):
        #  Run an update step
        trn_loss, trn_pred = fis.train(sess, trnData, trnLbls)
        # Evaluate on validation set
        val_pred, val_loss = fis.infer(sess, chkData, chkLbls)
        if epoch % 10 == 0:
            print("Train cost after epoch %i: %f" % (epoch, trn_loss))
        if epoch == num_epochs - 1:
            time_end = time.time()
            print("Elapsed time: %f" % (time_end - time_start))
            print("Validation loss: %f" % val_loss)
            # Plot real vs. predicted
            pred = np.vstack((np.expand_dims(trn_pred, 1),
                              np.expand_dims(val_pred, 1)))
            plt.figure("ANFIS Prediction")
            # plt.plot(ts)
            # plt.plot(pred)
            plt.plot(range(len(pred)),
                     pred, 'r', label='trained')
            plt.plot(range(len(lbls)), lbls, 'b', label='original')
            plt.legend(loc='upper left')
        trn_costs.append(trn_loss)
        val_costs.append(val_loss)
    # Plot the cost over epochs
    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.plot(np.squeeze(trn_costs))
    plt.title("Training loss, Learning rate =" + str(alpha))
    plt.subplot(2, 1, 2)
    plt.plot(np.squeeze(val_costs))
    plt.title("Validation loss, Learning rate =" + str(alpha))
    plt.ylabel('RMSE')
    plt.xlabel('Epochs')
    # Plot resulting membership functions
    fis.plotmfs(sess)
    plt.show()
