import numpy as np
import csv
from sklearn.datasets import fetch_mldata
from sklearn.metrics import classification_report,roc_curve

def onehotencoder(labels,start,stop):
    dist = 1 - start + stop
    size=labels.shape[0]
    mat = np.zeros((size,dist))
    for i in range(size):
        mat[i,int(labels[i])] = 1.0
    return mat

def mnistloader():
    mnist = fetch_mldata('MNIST original')
    data, labels = mnist["data"], mnist["target"]
    ltest,dtest = labels[60000:70000],data[60000:70000]
    ltest2=onehotencoder(ltest,0,9)
    ltrain,dtrain = labels[0:59999],data[0:59999]
    ltrain2=onehotencoder(ltrain,0,9)
    return ltrain2,dtrain,ltest2,dtest

train_labels, train_data, test_labels, test_data = mnistloader()


np.random.seed(8)
#size of neural network
hidden_layer_num = 3
training_size = 59999
test_size = 10000

input_nodes = train_data.shape[1]
hidden1_nodes = 500
hidden2_nodes = 300
hidden3_nodes = 100
output_nodes = 10

#hyperparameters
learning_rate = 0.000001
train_epoch_per_data = 30
reg_term = 0.01

def relu(x):
    y=x
    for i in range(x.shape[0]):
        y[i]=np.maximum(0,x[i])
    return y

def relu_derivative(x):
    y=x
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if(x[i][j]>0.0):
                y[i][j]= 1.0
            else:
                y[i][j]= 0.0
    return y


def softmax(x):
    exparr = np.exp(x-np.max(x))
    f = exparr/exparr.sum(axis=1)
    return f


def crossEntropy(output,labels):
    return (np.dot(labels,np.log(output)))

hid_layer1_weights = np.random.randn(input_nodes, hidden1_nodes)
hid_layer1_weights = hid_layer1_weights/(np.amax(hid_layer1_weights))
hid_layer1_biases = np.zeros((1,hidden1_nodes))
hid_layer2_weights = np.random.randn(hidden1_nodes ,hidden2_nodes)
hid_layer2_weights = hid_layer2_weights/(np.amax(hid_layer2_weights))
hid_layer2_biases = np.zeros((1,hidden2_nodes))
hid_layer3_weights = np.random.randn(hidden2_nodes ,hidden3_nodes)
hid_layer3_weights = hid_layer3_weights/(np.amax(hid_layer3_weights))
hid_layer3_biases = np.zeros((1,hidden3_nodes))
output_layer_weights = np.random.randn(hidden3_nodes ,output_nodes)
output_layer_weights = output_layer_weights/(np.amax(output_layer_weights))
output_layer_biases = np.zeros((1,output_nodes))
model_params = (hid_layer1_weights,hid_layer1_biases, hid_layer2_weights,hid_layer2_biases, hid_layer3_weights,hid_layer3_biases, output_layer_weights,output_layer_biases)


def feedforward(input_data,model_par):
    hlw_1,hlb_1, hlw_2,hlb_2, hlw_3,hlb_3, ow,ob = model_par
    m1 = np.dot(input_data ,hlw_1) + hlb_1
    z1 = relu(m1)
    m2 = np.dot(z1 ,hlw_2) + hlb_2
    z2 = relu(m2)
    m3 = np.dot(z2, hlw_3) + hlb_3
    z3 = relu(m3)
    m4 = np.dot(z3 ,ow) + ob
    predicted_prob=softmax(m4)
    
    return (predicted_prob,z1,z2,z3)


def backprop(input_data,input_label,transfer_model,model_par,epsilon):
    predicted_prob,z1,z2,z3 = transfer_model
    hlw_1,hlb_1, hlw_2,hlb_2, hlw_3,hlb_3, ow,ob = model_par
    
    #compute the derivative of the weights and biases and return the updated ones
    #d stands for derivative

    delta4 = (predicted_prob - input_label)  #predict - label
    dow = np.dot(z3.T,delta4)
    dob = np.sum(delta4)
    
    delta3 = np.multiply(np.dot(delta4,ow.T), relu_derivative(z3))
    
    dhlw_3 =np.dot(z2.T,delta3) 
    dhlb_3 =np.sum(delta3)
    
    delta2 = np.multiply(np.dot(delta3,hlw_3.T) , relu_derivative(z2))
    
    dhlw_2 =np.dot(z1.T,delta2) 
    dhlb_2 =np.sum(delta2)

    delta1 =np.multiply(np.dot(delta2,hlw_2.T) , relu_derivative(z1))
    
    dhlw_1 =np.dot(input_data.T,delta1) 
    dhlb_1 =np.sum(delta1)


    #regularization
    dhlw_1 += reg_term * hlw_1
    dhlw_2 += reg_term * hlw_2
    dhlw_3 += reg_term * hlw_3
    dow += reg_term * ow

    #updating
    hlw_1 -= epsilon*dhlw_1
    hlb_1 -= epsilon*dhlb_1
    hlw_2 -= epsilon*dhlw_2
    hlb_2 -= epsilon*dhlb_2
    hlw_3 -= epsilon*dhlw_3
    hlb_3 -= epsilon*dhlb_3
    ow -= epsilon*dow
    ob -= epsilon*dob
    
    return (hlw_1,hlb_1, hlw_2,hlb_2, hlw_3,hlb_3, ow,ob)


def train_nn(input_data,input_label,model_par,epsilon):
    #feedforward
    transfer_model_params=feedforward(input_data,model_par)   
    
    #backpropagation
    
    model_par=backprop(input_data,input_label,transfer_model_params,model_par,epsilon)
    
    return (model_par,transfer_model_params[0])


def predict_mnist(input_data,model_pars):
    x=feedforward(input_data,model_pars)
    return x[0]


def accuracy_calc(t_data,t_labels,model_pars,t_size):
    summation=0
    for i in range(t_size):
        if(np.argmax(predict_mnist(t_data[i],model_pars)) == np.argmax(t_labels[i])):
            summation+=1
    return ((100.0 * summation)/t_size)


def accuracy_calc_mse(t_data,t_labels,model_pars,t_size):
    summation=0
    for i in range(t_size):
        summation+=((predict_mnist(t_data[i],model_pars) - t_labels[i])** 2.0).mean(axis=1)
    return ((100.0 * summation)/t_size)


predicted = []
original = []

for j in range(training_size):
    for i in range(train_epoch_per_data):
        tr_data = train_data[j].reshape((1,train_data[j].shape[0]))
        model_params,pred = train_nn(tr_data,train_labels[j],model_params,learning_rate)
        tr_label = train_labels[j].reshape((1,train_labels[j].shape[0]))
        if (accuracy_calc_mse(tr_data,tr_label,model_params,1)==0.0):
            break        
    #calculate loss and print at regular intervals
    if j%1000==0:
        print("-------")
        print("Data :",j)
        print("Predicted_probabilities",pred[0])
        print("Actual Label",train_labels[j])
        #print(predicted,original)
    predicted.append(np.argmax(pred[0], axis=0))
    original.append(np.argmax(train_labels[j], axis=0))
        
print('Training accuracy:',accuracy_calc(train_data,train_labels,model_params,training_size),"%")
print('Test accuracy:',accuracy_calc(test_data,test_labels,model_params,test_size),"%")
print(classification_report(predicted,original))










