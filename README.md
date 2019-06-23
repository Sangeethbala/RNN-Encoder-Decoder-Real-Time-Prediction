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
