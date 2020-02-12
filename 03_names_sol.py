# -*- coding: utf-8 -*-

## http://www.nltk.org/book/ch06.html

import numpy as np
## names must be installed by running (e.g. interactively):
#import nltk
#nltk.download('names')
from nltk.corpus import names

from sklearn.feature_extraction import DictVectorizer

import sklearn
import tensorflow as tf
import numpy.random as random
random.seed(42)

import pylab as plt

def gender_features(word):
    return {'suffix1': word[-1:],
            'suffix2': word[-2:]}
# You can try to add trigrams, too:            
#            'suffix3': word[-3:]}

# You can use more precision, too:
#float_type = float_type

# Tensorflow standard is float32:
float_type = tf.float32

def build_graph(input_dim):
    ### define the model
    
    ## input
    x = tf.placeholder(float_type, (None, input_dim))
    y = tf.placeholder(float_type, (None,  1))
    
    ## simple model
    w = tf.Variable(tf.ones([input_dim, 1], dtype=float_type), name="weight", dtype=float_type)
    b = tf.Variable(tf.ones([1], dtype=float_type), name="bias", dtype=float_type)
    
    pred = tf.matmul(x, w) + b
    
    ## Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    
    # function must return x (placeholder), y (placeholder), pred (tensor), 
    # cost (op), optimizer (op) 
    return x, y, pred, cost, optimizer

if __name__ == "__main__":
    labeled_names = ([(name, 0) for name in names.words('male.txt')] +
    [(name, 1) for name in names.words('female.txt')])
    
    random.shuffle(labeled_names)
    
    featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
    train_set, test_set = featuresets[500:], featuresets[:500]
    
    train_feats = [namefeats[0] for namefeats in train_set]
    train_Y = [namefeats[1] for namefeats in train_set]
    train_Y = np.array(train_Y).reshape(len(train_Y), 1)
    
    test_feats = [namefeats[0] for namefeats in test_set]
    test_Y = [namefeats[1] for namefeats in test_set]
    test_Y = np.array(test_Y).reshape(len(test_Y), 1)
    
    feat_vectorizer = DictVectorizer(dtype=np.int32, sparse=False)
    
    train_X = feat_vectorizer.fit_transform(train_feats)
    test_X = feat_vectorizer.transform(test_feats)
    
    x, y, pred, cost, optimizer = build_graph(input_dim=test_X.shape[1])
    ## run everything
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        accs_test = []
        accs_train = []
        
        # Training cycle
        for epoch in range(6):
            # Run optimization op (backprop) and cost op (to get loss value)
            for x_in, y_out in zip(train_X, train_Y):
                _, c = sess.run([optimizer, cost], feed_dict={x: x_in.reshape(1, len(x_in)),
                                                              y: y_out.reshape(1,1)})
            
            train_pred = sess.run(tf.sigmoid(pred), {x: train_X})
            train_pred = [1 if elem > 0.5 else 0 for elem in train_pred]
            acc_train = sklearn.metrics.accuracy_score(train_pred, train_Y)
            print('Accuracy train:', acc_train)
            test_pred = sess.run(tf.sigmoid(pred), {x: test_X})
            test_pred = [1 if elem > 0.5 else 0 for elem in test_pred]
            acc_test = sklearn.metrics.accuracy_score(test_pred, test_Y)
            print('Accuracy test:', acc_test)
            print('loss:', c)
            accs_test.append(acc_test)
            accs_train.append(acc_train)

        plt.plot(accs_train)
        plt.plot(accs_test)

        plt.legend(["train accuracy", "test accuracy"])

        plt.show()

        test_name = 'El'

        gen_feat = gender_features(test_name)
        vec_feat = feat_vectorizer.transform(gen_feat)    
        print(vec_feat)

        myname_pred = sess.run(tf.sigmoid(pred), {x: vec_feat})

        print(myname_pred)

        print("Optimization Finished!")

