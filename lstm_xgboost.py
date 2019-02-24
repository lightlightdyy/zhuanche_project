import pandas as pd
import numpy as np
import datetime
from pandas import DataFrame

import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics
from scipy.stats import ranksums
from sklearn.model_selection import train_test_split

clean_neg_data_zc = pd.DataFrame(np.loadtxt('/data1/dengyuying/COMPETING_DATA/clean_neg_data_kc_1102.txt'))
clean_pos_data_zc = pd.DataFrame(np.loadtxt('/data1/dengyuying/COMPETING_DATA/clean_pos_data_kc.txt'))
if clean_pos_data_zc.shape[1] == 3601:
    clean_pos_data_zc = clean_pos_data_zc.iloc[:,1:]
if clean_neg_data_zc.shape[1] == 3601:
    clean_neg_data_zc = clean_neg_data_zc.iloc[:,1:]
#clean_neg_data_zc=clean_neg_data_zc.replace(-1,0)    
#clean_pos_data_zc=clean_pos_data_zc.replace(-1,0)    
response = [ 1 for _ in range(clean_pos_data_zc.shape[0])]
response += [0 for _ in range(clean_neg_data_zc.shape[0])]
yy = np.array(response)
xx = np.concatenate((clean_pos_data_zc, clean_neg_data_zc), axis=0)
X_train,X_test, y_train, y_test = train_test_split(xx,yy, test_size=0.4,random_state = 123)


DAYSINC = 30
shape_1=int(X_train.shape[1]/DAYSINC)
xxtrain=[]
for i in range(X_train.shape[0]):
    xxtrain+=list(X_train[i])
X_train=np.reshape(xxtrain,(X_train.shape[0],DAYSINC,shape_1))

xxtest=[]
for i in range(X_test.shape[0]):
    xxtest+=list(X_test[i])
X_test=np.reshape(xxtest,(X_test.shape[0],DAYSINC,shape_1))

"""
def standardize(train, test):
    #""" Standardize data """
# Standardize train and test
    X_train = (train - np.mean(train, axis=0)[None,:,:]) / np.std(train, axis=0)[None,:,:]
    X_test = (test - np.mean(test, axis=0)[None,:,:]) / np.std(test, axis=0)[None,:,:]
    return X_train, X_test
X_train, X_test = standardize(X_train, X_test)
"""    
training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 2947 testing series
n_steps = len(X_train[0])  # 128 timesteps per series
n_input = len(X_train[0][0])  # 9 input parameters per timestep

# LSTM Neural Network's internal structure
tf.reset_default_graph()  #为了更改模型参数
n_hidden = 80 # Hidden layer num of features   #original=32
n_classes = 2 # Total classes (should go up, or should go down)

# Training
learning_rate = 0.05  #0.005
lambda_loss_amount = 0.0015
training_iters = training_data_count * 10  # Loop 300 times on the dataset
batch_size = 1500
display_iter = 10000  # To show test set accuracy during training

# Some debugging info
print("Some useful info to get an insight on dataset's shape and normalisation:")
print("(X shape, y shape, every X's mean, every X's standard deviation)")
print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")

tf.reset_default_graph()  #为了更改模型参数
def LSTM_RNN(_X, _weights, _biases):
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    _X = tf.reshape(_X, [-1, n_input])
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
   # lstm_cell_3 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1,lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)
    lstm_last_output = outputs[-1]  
    # Linear activation
    #return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']
    
    output_feature = lstm_last_output =outputs[-2]  #输出倒数第二层特征 
    return tf.matmul(lstm_last_output, _weights['out'])+ _biases['out'],output_feature   #  【报错】



def extract_batch_size(_train, step, batch_size):
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)
    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index]
    return batch_s

def one_hot(y_, n_classes=n_classes):
    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS



x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Graph weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))  #mean=1.0
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


pred0 = LSTM_RNN(x, weights, biases)
pred=pred0[0]
feature=pred0[1]


# Loss, optimizer and evaluation
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
) # L2 loss prevents this overkill neural network to overfit the data
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2 # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# To keep track of training's performance
test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []

# Launch the graph
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

# Perform Training steps with "batch_size" amount of example data at each loop
step = 1
while step * batch_size <= training_iters:
    batch_xs = extract_batch_size(X_train, step, batch_size)
    batch_ys = one_hot(extract_batch_size(y_train, step, batch_size))

    # Fit training using batch data
    _, loss, acc = sess.run(
        [optimizer, cost, accuracy],
        feed_dict={
            x: batch_xs,
            y: batch_ys
        }
    )
    train_losses.append(loss)
    train_accuracies.append(acc)

    # Evaluate network only at some steps for faster training:
    if (step*batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):

        # To not spam console, show training accuracy/loss in this "if"
        print("Training iter #" + str(step*batch_size) + \
              ":   Batch Loss = " + "{:.6f}".format(loss) + \
              ", Accuracy = {}".format(acc))

        # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        loss, acc = sess.run(
            [cost, accuracy],
            feed_dict={
                x: X_test,
                y: one_hot(y_test)
            }
        )
        test_losses.append(loss)
        test_accuracies.append(acc)
        print("PERFORMANCE ON TEST SET: " + \
              "Batch Loss = {}".format(loss) + \
              ", Accuracy = {}".format(acc))

    step += 1

print("Optimization Finished!")




one_hot00 = sess.run(
    feature,
    feed_dict={
        x: X_test,
        y: one_hot(y_test)
    }
)

train = sess.run(
    feature,
    feed_dict={
        x: X_train,
        y: one_hot(y_train)
    }
)


train=pd.DataFrame(train)  #train=pd.DataFrame(np.load('train_fe.npy'))
test=pd.DataFrame(one_hot00)  #test=pd.DataFrame(np.load('test_fe.npy'))


#np.save('test_fe.npy',one_hot00)
#np.save('train_fe.npy',train)

#train=pd.read_csv('train_fe.csv')
#test=pd.read_csv('test_fe.csv')


import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import XGBRegressor   # this is for regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score
from scipy.stats import ranksums
from sklearn.metrics import precision_recall_fscore_support
import pickle
#xgb
x=np.array(pd.concat([train,test]))   
y=np.append(y_train,y_test)


def model(idx,train_data):
    r_auc = []
    r_acc = []
    r_pre = []
    r_rec = []
    for t in range(10):  # cross validation
        X, X_hold, Y, Y_hold = train_test_split(train_data, idx, test_size=0.5, random_state=t)
        pos_ind = np.where(Y == 1)[0]
        neg_ind = np.where(Y == 0)[0]
        pval = [ranksums(X[pos_ind, x],X[neg_ind, x]).pvalue for x in range(X.shape[1])]
        sorted_pval = sorted(pval)
        fe_num = min(80, len(sorted_pval)-1)
        index = np.where(pval <= sorted_pval[fe_num])[0]
    

        X = X[:,index]
        X_hold = X_hold[:,index]
        model = XGBClassifier(learning_rate =0.1, n_estimators=100, seed=t)
        model.fit(X, Y)
        importances = model.feature_importances_
        while any(importances == 0):
            X = X[:,importances>0]
            X_hold = X_hold[:,importances>0]
            model.set_params(n_estimators=np.sum(importances>0))
            model.fit(X, Y)
            importances = model.feature_importances_
        y_pred = model.predict_proba(X_hold)[:,1]
        predictions = [round(value) for value in y_pred]
        auc = roc_auc_score(Y_hold,y_pred)
        accuracy = accuracy_score(Y_hold, predictions)
        tmp = precision_recall_fscore_support(Y_hold,predictions)
        r_auc.append(auc)
        r_acc.append(accuracy)
        r_pre.append(tmp[0][1])
        r_rec.append(tmp[1][1])
        print(t, auc, accuracy, tmp[0][1], tmp[1][1])
    #Accuracy=np.mean(r_acc) * 100.0
    #AUC= np.mean(r_auc)
    #Precision=np.mean(r_pre)
    #Recall=np.mean(r_rec)
    #result=[Accuracy,AUC,Precision,Recall]
    print("Random Split Accuracy: %.2f%%" % (np.mean(r_acc) * 100.0))
    #print("Random Split Accuracy std: %.2f" % (np.std(r_acc) ))
    print("Random Split AUC: %.2f" % ( np.mean(r_auc) ))
    #print("Random Split AUC std: %.2f" % ( np.std(r_auc) ))
    print("Random Split Precision: %.2f" % ( np.mean(r_pre) ))
    print("Random Split Recall: %.2f" % ( np.mean(r_rec) ))
    
    # model building
    pos_ind = np.where(idx == 1)[0]
    neg_ind = np.where(idx == 0)[0]
    pval = [ranksums(train_data[pos_ind, x],train_data[neg_ind, x]).pvalue for x in range(train_data.shape[1])]
    sorted_pval = sorted(pval)
    fe_num = min(80, X.shape[1])
    
    index = np.where(pval <= sorted_pval[fe_num])[0]
    train_data = train_data[:,index]
    select_marker = np.array(range(train_data.shape[1]))
    model = XGBClassifier(learning_rate =0.1, n_estimators=100, seed=7)
    model.fit(train_data, idx )
    importances = model.feature_importances_
    while any(importances == 0):
        train_data = train_data[:,importances>0]
        select_marker = select_marker[importances>0]
        model.set_params(n_estimators=np.sum(importances>0))
        model.fit(train_data,idx)
        importances = model.feature_importances_ 
    model_path = './xgboost_10.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return importances, select_marker,  index, train_data
importances, select_marker,index, train_data = model(y, x)

