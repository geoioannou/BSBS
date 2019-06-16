from keras.datasets import boston_housing
import numpy as np
import tensorflow as tf
import keras
import sys

# Setting random seeds
rand_seed = 1
tf.set_random_seed(rand_seed)

# Download dataset
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(x_train.shape)
print(x_test.shape)

y_train = y_train.reshape((y_train.shape[0], 1))
y_test = y_test.reshape((y_test.shape[0], 1))


# Standardize the dataset
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)

x_train -=  mean
x_train /= std

x_test -=  mean 
x_test /= std



# Hyperparameters
batch_size = 40
lr = 0.001
epochs = 10
N = x_train.shape[0]
indices = np.arange(N)
num_batches = x_train.shape[0] // batch_size

# Number of Swapped Samples
swapped = 0
# Momentum Loss
ma_param = 0.6

# The file for saving the logs
logs = "./BostonLogs/swapped(0_60)"



# The simple fully connected model
X = tf.placeholder("float", shape=[None, 13])
Y = tf.placeholder("float", shape=[None, 1])
den = tf.layers.dense(X, 13)
out = tf.layers.dense(den, 1)

losses = tf.square(out - Y)

loss = tf.squeeze(losses, axis=1)
cost = tf.reduce_mean(loss)

step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)


# Summaries 
loss_summary = tf.summary.scalar('loss_per_batch', cost)
read_avg_batch = tf.placeholder(tf.float32, shape=())
batch_loss_summary = tf.summary.scalar('avg_batch_loss', read_avg_batch)
read_loss_tr = tf.placeholder(tf.float32, shape=())
loss_tr_summary = tf.summary.scalar('train_loss', read_loss_tr)
read_loss_test = tf.placeholder(tf.float32, shape=())
loss_test_summary = tf.summary.scalar('val_loss', read_loss_test)




sess = tf.InteractiveSession()
summ_writer = tf.summary.FileWriter(logs, sess.graph)

sess.run(tf.global_variables_initializer())




full_batch_losses = np.zeros(N)
EMA_batch_losses = np.zeros(N)

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

        _, c, losses, summ = sess.run([step, cost, loss, loss_summary], feed_dict = {X: batch_x, Y: batch_y})        
        
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



    # Train set Loss after each epoch
    los = 0
    for i in range(0, x_train.shape[0], 101):
        r_l = sess.run(loss, feed_dict = {X: x_train[i:i+101], Y: y_train[i:i+101]})
        los += r_l.sum()        
    los = los / x_train.shape[0]
    print(" Train Loss: ", los)
    
    summ2 = sess.run(loss_tr_summary, feed_dict = {read_loss_tr: los})
    summ_writer.add_summary(summ2, ep)


# Swapping Samples
    if swapped != 0:
        print("Swapping ", swapped, " samples.")
        ind_batch_low = np.argpartition(full_batch_losses, swapped)
        ind_ma_high = np.argpartition(EMA_batch_losses, N - swapped)
        
        batch_low_swap = ind_batch_low[swapped:]

        ma_high_swap = ind_ma_high[-swapped:]
        indices = np.concatenate((batch_low_swap, ma_high_swap))

# Change swapping number(optional)
    # swapped += int(60 / epochs)

    np.random.shuffle(indices)
    

# Test Loss after each epoch
    testLoss = sess.run(cost, feed_dict = {X: x_test, Y: y_test})
    print(" Test Loss: ", testLoss)
    summ4 = sess.run(loss_test_summary, feed_dict = {read_loss_test: testLoss})
    summ_writer.add_summary(summ4, ep)

    
