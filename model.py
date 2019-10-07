import numpy as np
import tensorflow as tf
import pickle
import math

data_fname = "word_data.pickle"
hidden_size = 150
embedding_size = 100
batch_size = 200
nepochs = 100000
learning_rate = 0.2
save_path = "models/simplernn_word.model"
save_rate = 1
predict_rate = 20

file_data = pickle.load(open(data_fname, "rb"))

# Arrange the data and maps
index_to_word = file_data["map"]
nclasses = len(index_to_word)
word_to_index = { index_to_word[c]:c for c in range(nclasses) }
data = np.array(file_data["data"], dtype=np.int32).T
input_size = output_size = nclasses
tsteps = data.shape[0]-1

# Prediction prompt
pred_input = ["no","i"]
pred_input_tsteps = len(pred_input)
pred_output_tsteps = tsteps

X = tf.placeholder(tf.int32, [ tsteps, batch_size ])
Y = tf.placeholder(tf.int32, [ tsteps, batch_size ])

X_onehot = tf.one_hot(X, input_size, axis=-1)
Y_onehot = tf.one_hot(Y, output_size, axis=-1)

# This function implements one forward pass through an RNN cell
def RNN(x, h_t1, name):
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    W = tf.get_variable("W", shape=[hidden_size,hidden_size], dtype=tf.float32, initializer=tf.random_normal_initializer)
    U = tf.get_variable("U", shape=[embedding_size,hidden_size], dtype=tf.float32, initializer=tf.random_normal_initializer)
    V = tf.get_variable("V", shape=[hidden_size,output_size], dtype=tf.float32, initializer=tf.random_normal_initializer)
    E = tf.get_variable("E", shape=[input_size,embedding_size], dtype=tf.float32, initializer=tf.random_normal_initializer)

    b = tf.get_variable("b", shape=[1,hidden_size], dtype=tf.float32, initializer=tf.zeros_initializer)
    c = tf.get_variable("c", shape=[1,output_size], dtype=tf.float32, initializer=tf.zeros_initializer)
    e = tf.get_variable("e", shape=[1,embedding_size], dtype=tf.float32, initializer=tf.zeros_initializer)

  # '@' is matrix multiplication
  e_t = tf.tanh(x @ E + e)

  a_t = b + h_t1 @ W + e_t @ U
  h_t = tf.tanh(a_t)
  y_hat_t = c + h_t @ V

  return (y_hat_t, h_t)

y_hat = []
h = tf.zeros([batch_size, hidden_size])

# Loop through each time step for training
for t in range(tsteps):
  y_hat_t, h = RNN(X_onehot[t], h, "layer1")
  y_hat.append(y_hat_t)

preds_X = tf.placeholder(tf.int32, [ pred_input_tsteps, 1 ])
preds_X_onehot = tf.one_hot(preds_X, input_size, axis=-1)

preds_y_hat_t = None
preds_h = tf.zeros([1, hidden_size])
for t in range(pred_input_tsteps):
  preds_y_hat_t, preds_h = RNN(preds_X_onehot[t], preds_h, "layer1")

preds_y_hat = []

# Loop through each time step for predictions
for t in range(pred_output_tsteps):
  preds_y_hat_t, preds_h = RNN(preds_y_hat_t, preds_h, "layer1")
  preds_y_hat.append(preds_y_hat_t)

preds_y_hat = tf.stack(preds_y_hat)

y_hat = tf.stack(y_hat) 
loss = tf.losses.softmax_cross_entropy(Y_onehot, y_hat) 

optimize = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

saver = tf.train.Saver()

losses = []

with tf.Session() as sess:
  try:
    saver.restore(sess, save_path)
  except:
    sess.run(tf.global_variables_initializer())

  # Run for nepochs, each with nbatches of size batch_size
  nbatches = int(data.shape[1] / batch_size)
  for e in range(nepochs):
    for b in range(nbatches):
      upper = (b+1)*batch_size
      if upper > data.shape[1]:
        upper = data.shape[1]
      batch_X = data[1:,(b*batch_size):upper]
      batch_Y = data[1:,(b*batch_size):upper]

      # Run the training step!
      _,l,x_one = sess.run((optimize, loss, X_onehot), feed_dict={X: batch_X, Y: batch_Y})
      losses.append(l)
  
      print("[Epoch " + str(e+1) + "/" + str(nepochs) + " Batch " + str(b) + "/" + str(nbatches) + "] Loss: " + str(l))

      # Predict every now and then. "pred_input" contains a list of word primers,
      # from which some more words are predicted by the model.
      if b % predict_rate == 0:
        preds = sess.run(preds_y_hat, feed_dict={preds_X: [ [word_to_index[l]] for l in pred_input ]}) 
        preds = np.argmax(preds, -1).flatten()
        preds = " ".join(pred_input) + " " + " ".join([ index_to_word[i] for i in preds ])
        print("Prediction: " + preds)
 
    saver.save(sess, save_path)
    print("Model saved to '" + save_path + "'")
    pickle.dump(losses, open("losses.pickle", "wb"))
