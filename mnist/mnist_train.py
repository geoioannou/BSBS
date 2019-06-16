import numpy as np
import tensorflow as tf
import sys

import keras
from keras.datasets import mnist

# Set random seed
rand_seed = 1
tf.set_random_seed(rand_seed)


img_rows, img_cols = 28, 28
batch_size = 128
num_classes = 10
epochs = 30

# Set the name of the file that stores the summaries for Tensorboard
logs = './yourFile'


# Download dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# Normalize the dataset to [0, 1]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



# Simple CNN model

X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
Y = tf.placeholder(tf.float32, shape=[None, 10])

training = tf.placeholder(tf.bool, shape=())


conv1 = tf.layers.conv2d(X, 32, kernel_size=3, activation=tf.nn.relu)
conv2 = tf.layers.conv2d(conv1, 64, kernel_size=3, activation=tf.nn.relu)

max1 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=(2, 2))

flat1 = tf.layers.flatten(max1)

den1 = tf.layers.dense(flat1, 128, activation=tf.nn.relu)
drop1 = tf.layers.dropout(den1, rate=0.50, training=training, seed=rand_seed)

out = tf.layers.dense(drop1, 10, activation=tf.nn.softmax)

_epsilon = tf.convert_to_tensor(tf.keras.backend.epsilon(), out.dtype.base_dtype)
out_clip = tf.clip_by_value(out, _epsilon, 1. - _epsilon)

loss = -(Y * tf.log(out_clip))

red_loss = tf.reduce_sum(loss, 1)

cost = tf.reduce_mean(red_loss)

step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

acct_mat = tf.equal(tf.argmax(out, 1), tf.argmax(Y, 1))
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))



# Summaries for keeping track of the training
loss_summary = tf.summary.scalar('loss_per_batch', cost)

read_avg_batch = tf.placeholder(tf.float32, shape=())
batch_loss_summary = tf.summary.scalar('avg_batch_loss', read_avg_batch)

read_loss_tr = tf.placeholder(tf.float32, shape=())
loss_tr_summary = tf.summary.scalar('train_loss', read_loss_tr)

read_loss_test = tf.placeholder(tf.float32, shape=())
loss_test_summary = tf.summary.scalar('val_loss', read_loss_test)

read_tr_acc = tf.placeholder(tf.float32, shape=())
tr_acc_summary = tf.summary.scalar('train_accuracy', read_tr_acc)

read_val_acc = tf.placeholder(tf.float32, shape=())
val_acc_summary = tf.summary.scalar('val_accuracy', read_val_acc)
# ------------------------------------------------



# Initialize the Session 
sess = tf.InteractiveSession()
summ_writer = tf.summary.FileWriter(logs, sess.graph)

sess.run(tf.global_variables_initializer())



N = x_train.shape[0]
indices = np.arange(N)
num_batches = x_train.shape[0] // batch_size

full_batch_losses = np.zeros(N)
EMA_batch_losses = np.zeros(N)


# BSBS Hyperparameters
swapped = 0
ma_param = 0.6


# Training Loop
for ep in range(epochs):
    print("Epoch: ", ep)
    
        
    x_train_temp = x_train[indices]
    y_train_temp = y_train[indices]

    avg_batch_perEp = 0
            
    for b in range(num_batches+1):
        if b == num_batches:
            batch_x = x_train_temp[b * batch_size :]
            batch_y = y_train_temp[b * batch_size :]
                
        else:
            batch_x = x_train_temp[b * batch_size : (b+1) * batch_size]
            batch_y = y_train_temp[b * batch_size : (b+1) * batch_size]

        _, c, losses, summ = sess.run([step, cost, red_loss, loss_summary], feed_dict = {X: batch_x, Y: batch_y, training: True})        
        
        summ_writer.add_summary(summ, ep * num_batches + b)            
        
        avg_batch_perEp = avg_batch_perEp + (c - avg_batch_perEp) / (b + 1)

        if b == num_batches:
            full_batch_losses[indices[b * batch_size : ]] = losses
            EMA_batch_losses[indices[b * batch_size : ]] = ma_param * EMA_batch_losses[indices[b * batch_size : ]] + (1 - ma_param) * losses

        else:
            full_batch_losses[indices[b * batch_size : (b+1) * batch_size]] = losses
            EMA_batch_losses[indices[b * batch_size : (b+1) * batch_size]] = ma_param * EMA_batch_losses[indices[b * batch_size : (b+1) * batch_size]] + (1 - ma_param) * losses

    summ_b_loss = sess.run(batch_loss_summary, feed_dict = {read_avg_batch: avg_batch_perEp})
    summ_writer.add_summary(summ_b_loss, ep)        
    print(" Avg Batch Loss: ", avg_batch_perEp)





# Measuring Train Loss and Accuracy at the end of the epoch
    los = 0
    s = 0
    for i in range(0, x_train.shape[0], 200):
        r_l, res = sess.run([red_loss, acct_res], feed_dict = {X: x_train[i:i+200], Y: y_train[i:i+200], training: False})
        los += r_l.sum()
        s += res
    los = los / x_train.shape[0]
    s = s / x_train.shape[0]
    print(" Train Loss: ", los)
    print(" Train Accuracy: ", s)

    
    summ2 = sess.run(loss_tr_summary, feed_dict = {read_loss_tr: los})
    summ_writer.add_summary(summ2, ep)

    summ3 = sess.run(tr_acc_summary, feed_dict = {read_tr_acc: s})
    summ_writer.add_summary(summ3, ep)




# Measuring Test Loss and Accuracy at the end of the epoch
    los = 0
    s = 0
    for i in range(0, x_test.shape[0], 200):
        r_l, res = sess.run([red_loss, acct_res], feed_dict = {X: x_test[i:i+200], Y: y_test[i:i+200], training: False})
        los += r_l.sum()
        s += res
    los = los / x_test.shape[0]
    s = s / x_test.shape[0]
    print(" Test Accuracy: ", s)
    summ4 = sess.run(loss_test_summary, feed_dict = {read_loss_test: los})
    summ_writer.add_summary(summ4, ep)

    summ5 = sess.run(val_acc_summary, feed_dict = {read_val_acc: s})
    summ_writer.add_summary(summ5, ep)
    


# Swapping samples
    if swapped != 0:
        print("Swapping ", swapped, " samples.")
        ind_batch_low = np.argpartition(full_batch_losses, swapped)
        
        ind_ma_high = np.argpartition(EMA_batch_losses, N - swapped)
        
        batch_low_swap = ind_batch_low[swapped:]

        ma_high_swap = ind_ma_high[-swapped:]

        indices = np.concatenate((batch_low_swap, ma_high_swap))


    # Optional if you want to change the number of swapped samples during training
    # swapped += int(9000 / epochs)

    np.random.shuffle(indices)
    


