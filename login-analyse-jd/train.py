import tensorflow as tf
from tensorflow.python.framework import ops
import DataManager

class RiskModel():
    def __init__(self):
        pass
    
    def create_placeholders(self, n_x, n_y):
        X = tf.placeholder(tf.float32, shape = (n_x, None))
        Y = tf.placeholder(tf.float32, shape = (n_y, None))
        
        return X, Y

    def initialize_parameters(self):

        #tf.set_random_seed(1) # so that your "random" numbers match ours 

        W1 = tf.get_variable("W1", [128,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b1 = tf.get_variable("b1", [128,1], initializer = tf.zeros_initializer())
        '''W2 = tf.get_variable("W2", [120, 125], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b2 = tf.get_variable("b2", [120, 1], initializer = tf.zeros_initializer())
        W3 = tf.get_variable("W3", [100, 120], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b3 = tf.get_variable("b3", [100, 1], initializer = tf.zeros_initializer())
        W4 = tf.get_variable("W4", [2, 100], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b4 = tf.get_variable("b4", [2, 1], initializer = tf.zeros_initializer())'''
        W2 = tf.get_variable("W2", [2, 128], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b2 = tf.get_variable("b2", [2, 1], initializer = tf.zeros_initializer())
        parameters = {"W1": W1, "b1": b1, "W2": W2,
                        "b2": b2}
        '''"W2": W2,
                        "b2": b2,
                        "W3": W3,
                        "b3": b3,
                        "W4": W4,
                        "b4": b4}'''
        
        return parameters

    def forward_propagation(self, X, parameters):
        """
        Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
        
        Arguments:
        X -- input dataset placeholder, of shape (input size, number of examples)
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                    the shapes are given in initialize_parameters

        Returns:
        Z3 -- the output of the last LINEAR unit
        """
        
        # Retrieve the parameters from the dictionary "parameters" 
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        '''W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']'''
            
        Z1 = tf.add(tf.matmul(W1, X), b1)                                              # Z1 = np.dot(W1, X) + b1
        A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)                                              # Z2 = np.dot(W2, a1) + b2
        '''A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)                                              # Z2 = np.dot(W2, a1) + b2
        A3 = tf.nn.relu(Z3)  
        Z4 = tf.add(tf.matmul(W4, A3), b4)                                              # Z3 = np.dot(W3,Z2) + b3
        '''
        
        return Z2

    def compute_cost(self, Z4, Y):
        """
        Computes the cost
        
        Arguments:
        Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
        Y -- "true" labels vector placeholder, same shape as Z3
        
        Returns:
        cost - Tensor of the cost function
        """
        
        # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
        logits = tf.transpose(Z4)
        labels = tf.transpose(Y)
        #logits = Z4
        #labels = Y
        
        ### START CODE HERE ### (1 line of code)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
        ### END CODE HERE ###
        
        return cost
    
    def train(self, X_train, Y_train, X_test=None, Y_test=None, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True, restore_params = False):
        
        ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
        tf.set_random_seed(1)                             # to keep consistent results
        seed = 3                                          # to keep consistent results
        (m, n_x) = X_train.shape
        n_y = Y_train.shape[1]
        costs = []
        X_train = X_train.T
        Y_train = Y_train.T
        X, Y = self.create_placeholders(n_x, n_y)

        parameters = self.initialize_parameters()

        Z4 = self.forward_propagation(X, parameters)
        
        cost = self.compute_cost(Z4, Y)
        
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
        
        # Initialize all the variables
        init = tf.global_variables_initializer()
        print("Start train")
        model_path = 'E:/Projects/python/kaggle-test/model/login-analyse-jd/model/simple.ckpt'
        saver = tf.train.Saver()
        if(restore_params):
            with tf.Session() as sess:
                # 读取之前训练好的数据
                load_path = saver.restore(sess, model_path)
                print("[+] Model restored from %s" % load_path)
                print('[+] Test accuracy is %f' % sess.run(accuracy, feed_dict={X: mnist.test.images, y_: mnist.test.labels}))
            return
        # Start the session to compute the tensorflow graph
        with tf.Session() as sess:
            
            # Run the initialization
            sess.run(init)
            
            # Do the training loop
            for epoch in range(num_epochs):

                epoch_cost = 0.                       # Defines a cost related to an epoch
                num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
                seed = seed + 1
                '''minibatches = (X_train, y_train)
                minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

                for minibatch in minibatches:

                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch
                    _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                    
                    epoch_cost += minibatch_cost / num_minibatches'''
                _ , epoch_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
                # Print the cost every epoch
                if print_cost == True and epoch % 100 == 0:
                    print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                if print_cost == True and epoch % 5 == 0:
                    costs.append(epoch_cost)

            # lets save the parameters in a variable
            parameters = sess.run(parameters)
            print ("Parameters have been trained!")
            #save_path = saver.save(sess, model_path)
            #print("[+] Model saved in file: %s" % save_path)

            # Calculate the correct predictions
            correct_prediction = tf.equal(tf.argmax(Z4), tf.argmax(Y))

            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
            #print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
            
            return parameters
if __name__ == "__main__":
    tfRecordManager = DataManager.TFRecordManager()
    datas, labels = tfRecordManager.readTf_simple("E://Projects//python//kaggle-test//login-analyse-jd//data//tf-train-all")
    print("Shape Data {}, labels {}".format(datas.shape, labels.shape))
    riskModel = RiskModel()
    riskModel.train(datas, labels)