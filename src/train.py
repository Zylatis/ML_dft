import tensorflow as tf
import numpy as np
import random
import datetime
import sklearn.model_selection as sk
from sklearn.preprocessing import StandardScaler

def get_accuracy( exact, predicted ):
	tol = 0.05
	n = len(exact)
	perc_diff = [ abs((exact[i] - predicted[i])/exact[i]) for i in range(n) ]
	n_accurate = 0
	for val in perc_diff:
		if val < tol:
			n_accurate = n_accurate+1
			
	return 1.*n_accurate/n

n_threads=4

with open('../train_data/basis_pars.dat', 'r') as ins:
    density_data = [[float(n) for n in line.split()] for line in ins]

# Herp de derp - need minus sign here because it cannot fit on a negative number with given activation functions!
with open('../train_data/energies.dat', 'r') as ins:
    energy_data = [ -float(n) for n in ins ]


density_data = density_data[:1250]
energy_data = energy_data[:1250]

density_train, density_test, energy_train, energy_test = sk.train_test_split(density_data,energy_data,test_size=0.10 )

# scaler = StandardScaler().fit(density_train)

# density_train = scaler.transform(density_train)
# density_test = scaler.transform(density_test)

density_train = np.asarray(density_train)
energy_train = np.asarray(energy_train)

density_test = np.asarray(density_test)
energy_test = np.asarray(energy_test)

# Python optimisation variables
learning_rate = 0.01
epochs = pow(10,8)
batch_size = 25
n_train = len(energy_train)
n_test = len(density_data) - n_train
nR = len(density_train[0])
n_nodes = 50

print( "Training size: ", n_train )
print( "Test size: ",  n_test )
print( "Input size (nR): ", nR)
print( "Hidden layer size: ", n_nodes) 
print( "Learning rate: ", learning_rate )
print( "Batch size: ", batch_size)
print( "Epochs: ", epochs )

with tf.name_scope("Input"):
	x = tf.placeholder(tf.float32, [None, nR], name = "x")
with tf.name_scope("Output"):
	y = tf.placeholder(tf.float32, [None], name = "y")

with tf.name_scope("Hidden_layer"):
	# W1 = tf.Variable(tf.random_normal([nR,n_nodes], mean = 0.0, stddev = 0.1), name='W1')
	W1 = tf.get_variable("W1", shape=[nR, n_nodes],initializer=tf.contrib.layers.xavier_initializer())
	b1 = tf.Variable(1., name='b1')

with tf.name_scope("Hidden_layer2"):
	# W1 = tf.Variable(tf.random_normal([nR,n_nodes], mean = 0.0, stddev = 0.1), name='W1')
	W12 = tf.get_variable("W12", shape=[n_nodes, n_nodes],initializer=tf.contrib.layers.xavier_initializer())
	b12 = tf.Variable(1., name='b12')


with tf.name_scope("Output_layer"):
	# W2 = tf.Variable(tf.random_normal([n_nodes,1], mean = 0.0, stddev = 0.1), name='W2')
	W2 = tf.get_variable("W2", shape=[n_nodes, 1],initializer=tf.contrib.layers.xavier_initializer())
	b2 = tf.Variable(1., name='b2')


with tf.name_scope("Hidden_layer_computation"):
	# calculate the output of the hidden layer
	hidden_out = tf.add(tf.matmul(x, W1), b1)
	hidden_out = tf.nn.leaky_relu(hidden_out,0.1)

 # keep_prob = tf.placeholder("float")
  	# hidden_layer_drop = tf.nn.dropout(hidden_out, 0.1)  


with tf.name_scope("Output_layer_computation"):
	y_ = tf.add(tf.matmul(hidden_out,W2), b2)
	y_ = tf.reshape(y_, [tf.size(y)])
	y_ = tf.nn.leaky_relu(y_, 0.1)
	#~ y_ = tf.nn.softplus(y_)
#~ tf.reduce_mean(tf.square(output - target))
with tf.name_scope("Cost"):
	cost =  tf.reduce_sum(pow(y-y_,2))
	# ~ mean_accuracy = 100.*tf.reduce_mean(abs(y-y_)/y)
	#~ mean_accuracy = 100.*get_accuracy(y.eval(), y_.eval())
	#~ max_accuracy = 100.*tf.reduce_max(abs(y-y_)/y)

with tf.name_scope("Train"):
# add optimizer
	optimiser = tf.train.AdamOptimizer( learning_rate ).minimize(cost)
	
	
# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

saver = tf.train.Saver()
#~ tf.summary.scalar("accuracy", mean_accuracy)	
tf.summary.histogram("W1",W1)
tf.summary.histogram("W2",W2)
tf.summary.histogram("b1",b1)
tf.summary.histogram("b2",b2)

tf.add_to_collection('vars', W1)
tf.add_to_collection('vars', W2)
tf.add_to_collection('vars', b1)
tf.add_to_collection('vars', b2)
config = tf.ConfigProto(intra_op_parallelism_threads = n_threads, inter_op_parallelism_threads = n_threads, allow_soft_placement = True)

#~ # start the session
with tf.Session(config=config) as sess:
	sess.run(init_op)
	
	summaryMerged = tf.summary.merge_all()
	#~ filename="/summary_log/run"+datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%s")
	#~ filename = "summary_log/model_graph" 
	#~ writer = tf.summary.FileWriter(filename, sess.graph)
	
	for epoch in range(epochs):
		batch_pos = random.sample(range(0,n_train), batch_size)
		with tf.name_scope("Batch_selection"):
			batch_x = density_train[batch_pos]
			batch_y = energy_train[batch_pos]
		
		_, c,sumOut = sess.run([optimiser, cost, summaryMerged], feed_dict={x: batch_x, y: batch_y})
		train_cost = sess.run(cost, feed_dict={x: density_train, y: energy_train})
		test_cost = sess.run(cost, feed_dict={x: density_test, y: energy_test})
		
		train_predict = sess.run(y_, feed_dict={x: density_train, y: energy_train})
		test_predict = sess.run(y_, feed_dict={x: density_test, y: energy_test})
		
		train_acc = get_accuracy(energy_train, list(train_predict))
		test_acc = get_accuracy(energy_test, list(test_predict))
	
		#~ print  sess.run(tf.shape(hidden_out), feed_dict={x: batch_x, y: batch_y})
		#~ writer.add_summary(sumOut,epoch)
		if(epoch%100 == 0):
			print(epoch, round(train_cost,2), "			", round(test_cost, 2),"			", round( train_acc, 2),"			", round(test_acc, 2))
		if(train_acc > 0.95):
			# print te, test_acc
			#~ print (epoch, )#, val_cost
			break
		
	#~ save_path = saver.save(sess, "model_save/model.ckpt")
	

	train_predict = sess.run(y_, feed_dict={x: density_train, y: energy_train})
	test_predict = sess.run(y_, feed_dict={x: density_test, y: energy_test})

print test_predict
print energy_test
all_predict = -1.*np.asarray( list(train_predict) + list(test_predict) )
np.savetxt( "../predicted_data/predicted_energies.dat" , all_predict )
print( "-------------------------")
#~ n_dat = len(energy_train)
#~ with tf.Session() as sess:
	#~ saver.restore(sess, "/home/graeme/Dropbox/MaitraGroup/side_quests/ML/X.ckpt")
	#~ batch_x = density_train_val
	#~ batch_y = energy_train_val
	#~ cost_calc = sess.run(cost, feed_dict={x: density_train_val, y: energy_train_val})
	#~ print sess.run(y_, feed_dict={x: density_train_val, y: energy_train_val})
	#~ print "Validated energy_train: ", 100*cost_calc
