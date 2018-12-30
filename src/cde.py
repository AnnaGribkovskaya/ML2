from NeuralNetwork import NeuralNetwork, LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from a import ising_energies, generate_data_set
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.sparse as sp
import pickle
import seaborn 
import warnings
import os

#NEW LIFE!!!!
# system size
L=40
n_samples=600
# create 10000 random Ising states
states=np.random.choice([-1, 1], size=(10000,L))
epochs = 200
batch_size = 32
n_hidden_neurons = 256
n_categories = 1
activationFunctionHidden='relu' 
activationFunctionOutput='linear'

etas = np.logspace(-5, -3, 3)
lmbdas = np.logspace(-3,1, 5)

r2TrainMatrix = np.zeros((len(etas), len(lmbdas)))
r2TestMatrix = np.zeros_like(r2TrainMatrix)
mseTrainMatrix = np.zeros((len(etas), len(lmbdas)))
mseTestMatrix = np.zeros_like(r2TrainMatrix)

models = np.zeros((len(etas), len(lmbdas)), dtype=object)
print('Generate data set')
data, X_train, X_test, Y_train, Y_test = generate_data_set(states,L,n_samples)
print('Start Training NN')
for i, eta in enumerate(etas):
    for j, lmbd in enumerate(lmbdas):
        
        dnn = NeuralNetwork(X_train, Y_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                            n_hidden_neurons=n_hidden_neurons, n_categories=n_categories, 
                           activationFunctionHidden=activationFunctionHidden, \
                            activationFunctionOutput=activationFunctionOutput)
        dnn.train()
        models[i][j] = dnn
        #dnn.train2()
        
        train_predict = dnn.predict(X_train)
        mseTrain = mean_squared_error(Y_train, train_predict)
        mseTrainMatrix[i][j] = mseTrain
        r2Train = r2_score(Y_train, train_predict)
        r2TrainMatrix[i][j] = r2Train
        
        test_predict = dnn.predict(X_test)
        mseTest = mean_squared_error(Y_test, test_predict)
        mseTestMatrix[i][j] = mseTest
        r2Test = r2_score(Y_test, test_predict)
        r2TestMatrix[i][j] = r2Test



print('Start plotting:')
''' PLOTS 
x = [str(i) for i in etas]
y = [str(i) for i in lmbdas]

fontsize = 15
fig, (ax, ax2) = plt.subplots(1,2, figsize=(12.5,5))
seaborn.heatmap(r2TrainMatrix.T, annot=True, ax=ax, cmap='Greys')
ax.set_title('Training set')
ax.set_xticklabels(x) 
ax.set_yticklabels(y)
ax.set_xlabel('$\\mathrm{learning\\ rate}$',fontsize=fontsize)
ax.set_ylabel('$\\mathrm{regularization\\ }$',fontsize=fontsize)

seaborn.heatmap(r2TestMatrix.T, annot=True, ax=ax2, cmap='Greys')
ax2.set_xticklabels(x)
ax2.set_yticklabels(y)
ax2.set_xlabel('$\\mathrm{learning\\ rate}$',fontsize=fontsize)
ax2.set_ylabel('$\\mathrm{regularization\\ }$',fontsize=fontsize)
ax2.set_title('Test set')

plt.tight_layout()

plt.draw()
fig.savefig("R2forNeurons" + str(n_hidden_neurons) + ".pdf", bbox_inches='tight')
'''

#READ data from METHA

def read(t=0.25,root="./"):
    if t > 0.:
        data = pickle.load(open(root+'Ising2DFM_reSample_L40_T=%.2f.pkl'%t,'rb'))
    else:
        data = pickle.load(open(root+'Ising2DFM_reSample_L40_T=All.pkl','rb'))
    #data = np.unpackbits(data).astype(int).reshape(-1,1600)
    data = np.int8(np.unpackbits(data).reshape(-1,1600))
    data[np.where(data==0)]=-1
    return data

cwd = os.getcwd()
root=cwd + '/IsingData/'#os.path.expanduser('~')+'IsingData/'
stack = []
labels = np.zeros((10000*13))
#labels = np.zeros((10000*11))
counter = 0
for t in .25, .5, .75, 1., 1.25, 1.5, 1.75, 2.75, 3., 3.25, 3.5, 3.75, 4.0:#np.arange(0.25,4.01,0.25):
#for t in .5, .75, 1., 1.25, 1.5, 1.75, 2.75, 3., 3.25, 3.5, 3.75:
    stack.append(read(t, root=root))
    y = np.ones(10000,dtype=int)
    if t > 2.25:
        y*=0

    labels[counter*10000:(counter+1)*10000] = y
    counter += 1
data = np.vstack(stack)
del stack

num_classes=2
train_to_test_ratio=0.5 # training samples
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(data,labels,train_size=train_to_test_ratio, \
                                              test_size = 1-train_to_test_ratio)

epochNumbers = [30, 50, 70]
epochSGDnumbers = [50, 100, 150]
t1s = [5, 25, 50]
etas = np.logspace(-2,0, 3)
iterationNumbers = [50, 100, 150]

lr2 = LogisticRegression(X_train,Y_train)
lr2.createDesignMatrix()

accuracyTrainMiniBatch = np.zeros((len(epochNumbers), len(t1s)))
accuracyTestMiniBatch = np.zeros((len(epochNumbers), len(t1s)))
accuracyTrainStandardGD = np.zeros((len(iterationNumbers), len(etas)))
accuracyTestStandardGD = np.zeros((len(iterationNumbers), len(etas)))
accuracyTrainSGD = np.zeros((len(epochSGDnumbers), len(t1s)))
accuracyTestSGD = np.zeros((len(epochSGDnumbers), len(t1s)))

print('LogReg start: ')
for i, epoch, epochSGD, iterationNumber in zip(range(len(epochNumbers)), epochNumbers, epochSGDnumbers, iterationNumbers):
    for j, eta, t1 in zip(range(len(etas)), etas, t1s):

        # Mini-batch stochastic gradient descent
        betaMB, costMB, betaBestMB = lr2.stochasticGradientDescent2(nEpochs=epoch, batchSize=20,t0=5, t1 = t1)
        myPredictHardTrain = lr2.predictHard(X_train)
        accuracyMyModelTrain = (myPredictHardTrain == Y_train).mean()
        print('\nAccuracy mini-batch, train', accuracyMyModelTrain)
        myPredictHardTest = lr2.predictHard(X_test)
        accuracyMyModelTest = (myPredictHardTest == Y_test).mean()
        print('Accuracy mini-batch, test', accuracyMyModelTest)
        print('Best cost: ', np.min(costMB))
        accuracyTrainMiniBatch[i,j] = accuracyMyModelTrain
        accuracyTestMiniBatch[i,j] = accuracyMyModelTest

        # Gradient descent
        lr2.gradientDescent(iterations=iterationNumber,tolerance = 1e-20, eta=eta)
        myPredictHardTrain = lr2.predictHard(X_train)
        accuracyMyModelTrain = (myPredictHardTrain == Y_train).mean()
        print('\nAccuracy Gradient descent, train', accuracyMyModelTrain)
        myPredictHardTest = lr2.predictHard(X_test)
        accuracyMyModelTest = (myPredictHardTest == Y_test).mean()
        print('Accuracy Gradient descent, test', accuracyMyModelTest)
        accuracyTrainStandardGD[i,j] = accuracyMyModelTrain
        accuracyTestStandardGD[i,j] = accuracyMyModelTest

        # Stochastic gradient descent
        betaS, costS, betaBestS = lr2.stochasticGradientDescent( tolerance = 1e-8, n_epochs=epochSGD,t0=5, t1 = t1)
        myPredictHardTrain = lr2.predictHard(X_train)
        accuracyMyModelTrain = (myPredictHardTrain == Y_train).mean()
        print('\nAccuracy Stochastic gradient descent, train', accuracyMyModelTrain)
        myPredictHardTest = lr2.predictHard(X_test)
        accuracyMyModelTest = (myPredictHardTest == Y_test).mean()
        print('Accuracy Stochastic gradient descent, test', accuracyMyModelTest)
        print('Best cost: ', np.min(costS))
        accuracyTrainSGD[i,j] = accuracyMyModelTrain
        accuracyTestSGD[i,j] = accuracyMyModelTest



''' TESTS AND PLOTS

x=[str(i) for i in etas]
y=[str(i) for i in iterationNumbers]

fig, (ax, ax2) = plt.subplots(1,2, figsize = (13.5,5))
seaborn.heatmap(accuracyTrainStandardGD.T, annot=True, ax=ax, cmap='Greys')
ax.set_title('GD Train accuracy')
ax.set_xticklabels(x) 
ax.set_yticklabels(y)
ax.set_xlabel('$\\mathrm{learning\\ rate}$',fontsize=fontsize)
ax.set_ylabel('$\\mathrm{iterations}$',fontsize=fontsize)

seaborn.heatmap(accuracyTestStandardGD.T, annot=True, ax=ax2, cmap='Greys')
ax2.set_xticklabels(x)
ax2.set_yticklabels(y)
ax2.set_xlabel('$\\mathrm{learning\\ rate}$',fontsize=fontsize)
ax2.set_ylabel('$\\mathrm{iterations}$',fontsize=fontsize)
ax2.set_title('GD Test accuracy')

plt.tight_layout()
plt.draw()
fig.savefig("LogRegGD.pdf", bbox_inches='tight')

x=[str(i) for i in t1s]
y=[str(i) for i in epochSGDnumbers]

fig, (ax, ax2) = plt.subplots(1,2, figsize = (13.5,5))
seaborn.heatmap(accuracyTrainSGD.T, annot=True, ax=ax, cmap='Greys')
ax.set_title('SGD Training accuracy')
ax.set_xticklabels(x) 
ax.set_yticklabels(y)
ax.set_xlabel('$\\mathrm{t1}$',fontsize=fontsize)
ax.set_ylabel('$\\mathrm{epochs}$',fontsize=fontsize)

seaborn.heatmap(accuracyTestSGD.T, annot=True, ax=ax2, cmap='Greys')
ax2.set_xticklabels(x)
ax2.set_yticklabels(y)
ax2.set_xlabel('$\\mathrm{t1}$',fontsize=fontsize)
ax2.set_ylabel('$\\mathrm{epochs}$',fontsize=fontsize)
ax2.set_title('SGD Test accuracy')

plt.tight_layout()
plt.draw()
fig.savefig("LogRegSGD.pdf", bbox_inches='tight')

x=[str(i) for i in t1s]
y=[str(i) for i in epochNumbers]

fig, (ax, ax2) = plt.subplots(1,2, figsize = (13.5,5))
seaborn.heatmap(accuracyTrainMiniBatch.T, annot=True, ax=ax, cmap='Greys')
ax.set_title('Mini-batch Training accuracy')
ax.set_xticklabels(x) 
ax.set_yticklabels(y)
ax.set_xlabel('$\\mathrm{t1}$',fontsize=fontsize)
ax.set_ylabel('$\\mathrm{epochs}$',fontsize=fontsize)

seaborn.heatmap(accuracyTestMiniBatch.T, annot=True, ax=ax2, cmap='Greys')
ax2.set_xticklabels(x)
ax2.set_yticklabels(y)
ax2.set_xlabel('$\\mathrm{t1}$',fontsize=fontsize)
ax2.set_ylabel('$\\mathrm{epochs}$',fontsize=fontsize)
ax2.set_title('Mini-batch Test accuracy')

plt.tight_layout()
plt.draw()
fig.savefig("LogMinBatch.pdf", bbox_inches='tight')

#plt.tight_layout()
#plt.show()



def test_NN():
    
    np.random.seed(1)
    from sklearn.datasets import make_regression
    from sklearn.preprocessing import MinMaxScaler

    X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=1)
    scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
    scalarX.fit(X)
    scalarY.fit(y.reshape(100,1))
    X = scalarX.transform(X)
    y = scalarY.transform(y.reshape(100,1))

    epochs = 1000
    batch_size = 10
    eta = 0.01
    lmbd = 0# 0.01
    n_hidden_neurons = 4
    n_categories = 1
    activationFunctionHidden='relu' #'sigmoid'
    activationFunctionOutput='linear'

    dnn = NeuralNetwork(X, y, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                        n_hidden_neurons=n_hidden_neurons, n_categories=n_categories, 
                       activationFunctionHidden=activationFunctionHidden, activationFunctionOutput=activationFunctionOutput)
    dnn.train()
    Xnew, a = make_regression(n_samples=3, n_features=2, noise=0.1, random_state=1)
    Xnew = scalarX.transform(Xnew)
    test_predict = dnn.predict(Xnew)
    print('Own class:')
    for i in range(len(Xnew)):
        print("X=%s, Predicted=%s" % (Xnew[i], test_predict[i]))


    # Keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    model = Sequential()
    model.add(Dense(n_hidden_neurons, input_dim=2, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='sgd')
    model.fit(X, y, epochs=1000, verbose=0)
    ynew = model.predict(Xnew)
    # show the inputs and predicted outputs
    print('\nKeras:')
    for i in range(len(Xnew)):
        print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
    tolerance = 0.2   
    success = np.max(abs(np.divide(test_predict,ynew)-1)) < tolerance
    msg = 'Max ratio prediction own class to Keras ',  np.max(abs(np.divide(test_predict,ynew)-1))
    assert success, msg
test_NN()

