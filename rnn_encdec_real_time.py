#
# RNN Encoder Decoder for real-time time series prediction. The code is offline
# implementation but sequential(imitating real time). As each time series sample
# comes into the network, RNN predicts the 10th(user defined) time step ahead.
# This is especially useful for many applications like real time dynamic
# prediction events like collision, explosion etc.
# Some parts of the code were inspired from the offline presentation of the
# RNN encoder decoder in this blog: https://weiminwang.blog/2017/09/29/multivariate-time-series-forecast-using-seq2seq-in-tensorflow/
# Used many to many model to account the dependence of time
# steps on previous time steps. Using many to many Neural Network model,
# we get a number of values(determined by user) for a single acceleration value.
# Man value gave us improved prediction results for our high rate sampled data.
# Parts of this code like filename, input data size etc needs to be modified
# according to the data user considers.
#
import numpy as np
import matplotlib.pyplot as plt
from plant import plant
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
import math
from scipy.io import loadmat
#
# Find the Mean and Standard Deviation.
#
def average_adiag(x):
    x1d = [np.mean(x[::-1, :].diagonal(pp)) for pp in
           range(-x.shape[0] + 1, x.shape[1])]
    st1d = [np.std(x[::-1, :].diagonal(dd)) for dd in
            range(-x.shape[0] + 1, x.shape[1])]
    x1d = np.array(x1d)
    st1d = np.array(st1d)
    return x1d, st1d
# Import file and get data.
fname          = 'input.mat' # Acceleration file name. Change accordingly.
data           = loadmat(fname)
header         = 1
lines          = data['input']
data_inp = np.zeros((len(lines), header))
for i, line in enumerate(lines):
    values         = line
    data_inp[i, :] = values

lookback_size  = 50 # Number of time steps to look back from the current time step.
timesteps      = 500 # Number of time steps to process or total number of samples.
x_data_scaled  = np.zeros((1, lookback_size, 1)) # Initialize x_data

actual_out_all = []
pred_outs_all  = []
#
# For our data, we just consider acceleration after 2000th time step only.
# Change accordingly per the data you are considering.
#
data_inp = data_inp[2000:]
#
# Putting in range -1 to +1
#
scaler   = MinMaxScaler(feature_range = (-1, 1))
scaler   = scaler.fit(data_inp)
data_inp = scaler.transform(data_inp).T
data_inp = data_inp.reshape(2000,)
#
# Here we are predicting the acceleration output 10 time steps after the current time step.
#
input_data_scaled  = data_inp[:-10]
output_data_scaled = data_inp[10:]

input_data_scaled  = input_data_scaled.reshape(input_data_scaled.shape[0], )
output_data_scaled = output_data_scaled.reshape(output_data_scaled.shape[0],)

from build_model_basic1 import *

# length of input signals
input_seq_len  = lookback_size
# length of output signals
output_seq_len = 5
# Hidden layer size of LSTM Cell
hidden_dim     = 5
# num of input signals
input_dim      = 1
# num of output signals
output_dim     = 1
y_data_scaled  = np.zeros((1, output_seq_len,))

num_iterations = 5 # Similar to epochs in offline learning
rnn_enc_model  = build_graph(feed_previous = False)
init           = tf.global_variables_initializer()
start_time     = time.time()
with tf.Session() as sess:
    g = 0
    sess.run(init)
    for m in range(timesteps - output_seq_len):
        #
        # Start after lookback_size size.
        #
        if m > 50 + lookback_size:
            start_idx = int(m - lookback_size)
            end_idx   = min(m, timesteps)
            x_data_scaled[0] = input_data_scaled[start_idx : end_idx].reshape(end_idx - start_idx, 1)
            y_data_scaled[0] = output_data_scaled[m - 1 : m + output_seq_len - 1,]

            g = g + 1
            #
            # First predict the variable value 10 time steps ahead and then train the network. For the first
            # time step, only train no prediction.
            #
            if g != 1:
                feed_dict = {rnn_enc_model['enc_inp'][t]: x_data_scaled[0][t].reshape(-1, input_dim) for t in range(input_seq_len)}
                feed_dict.update({rnn_enc_model['target_seq'][t]: np.zeros([1, output_dim]) for t in range(output_seq_len)})
                feed_dict.update({rnn_enc_model['feed_previous'][0]: 'True'}) #for t in range(output_seq_len)})
                pred_outs_scaled  = sess.run(rnn_enc_model['reshaped_outputs'], feed_dict)
                actual_out_scaled = y_data_scaled[0][0].reshape(1,1)
                
                pred_outs  = scaler.inverse_transform(np.asarray(pred_outs_scaled).reshape(np.asarray(pred_outs_scaled).shape[0], 1))
                actual_out = scaler.inverse_transform(actual_out_scaled.reshape(actual_out_scaled.shape[0], 1))
                actual_out_all.append(actual_out)
                pred_outs_all.append(pred_outs)
            #
            # Train the network for num_iterations.
            #
            for i in range(num_iterations):
                feed_dict = {rnn_enc_model['enc_inp'][t]: x_data_scaled[0][t].reshape(-1,input_dim) for t in range(input_seq_len)}
                feed_dict.update({rnn_enc_model['target_seq'][t]: y_data_scaled[0][t].reshape(-1,output_dim) for t in range(output_seq_len)})
                feed_dict.update({rnn_enc_model['feed_previous'][0]: 'False'})
                _, loss_t = sess.run([rnn_enc_model['train_op'], rnn_enc_model['loss']], feed_dict)

            print("Timestep {0} ".format(m + 1))
            
print("--- %s seconds ---" % (time.time() - start_time))
#
# For each time step we get output_seq_len number of values. Find out the mean and standard deviation of
# outputs at each time step.
#
pred_outputs = np.zeros(shape = (len(pred_outs_all), output_seq_len))
for n in range(len(pred_outs_all)):
    pred_outputs[n, :] = pred_outs_all[n].T
 
mean_pred_out, std_pred_out = average_adiag(pred_outputs)
pred_outs_all               = mean_pred_out[0:len(mean_pred_out) - (output_seq_len - 1)]
pred_outs_all_std           = std_pred_out[0:len(std_pred_out) - (output_seq_len- 1)]
times                       = np.zeros((pred_outs_all.shape[0], 1))
for i in range(times.shape[0] - 1):
    times[i] = i * .0001
# Plot Fitted vs Predicted curves
times          = times.reshape(times.shape[0], )
pred_outs_all  = np.array(pred_outs_all)
pred_outs_all  = pred_outs_all.flatten().reshape(pred_outs_all.shape[0],)
plt.xlabel("Time", fontsize = 25)
plt.ylabel("Acceleration", fontsize = 25)
actual_out_all = np.array(actual_out_all)
actual_out_all = actual_out_all.flatten().reshape(actual_out_all.shape[0], 1)
plt.plot(times, pred_outs_all, label = 'Fitted : Mean')
plt.plot(times, actual_out_all, label = 'True')
plt.legend()
plt.show()
# RMSE
Error  = (pred_outs_all - actual_out_all)**2
print("RMSE = " + str(sqrt(np.mean(np.abs(Error)))))
