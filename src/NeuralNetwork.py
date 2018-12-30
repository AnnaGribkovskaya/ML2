import numpy as np

# Based on the code from lecture notes 

class NeuralNetwork:

    def __init__(
        self,
        X_data,
        Y_data,
        n_hidden_neurons=50,
        n_categories=10,
        epochs=10,
        batch_size=100,
        eta=0.1,
        lmbd=0.0,
        #Additionly have a default values:
        activationFunctionHidden='sigmoid',
        activationFunctionOutput='linear'
    ):
        
        
        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        if X_data.ndim == 1:
            self.n_features = 1
        else:
            self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd
        

        if activationFunctionHidden=='sigmoid':
            self.activationFunctionHidden = self.sigmoid
            self.derivativeHidden = self.sigmoidDerivative

        if activationFunctionHidden=='relu':
            self.activationFunctionHidden = self.relu
            self.derivativeHidden = self.reluDerivative

        
        if activationFunctionOutput == 'linear':
            self.activationFunctionOutput = self.linear
            self.derivativeOutput = self.linearDerivative
    
        if activationFunctionOutput == 'sigmoid':
            self.activationFunctionOutput = self.sigmoid
            self.derivativeOutput = self.sigmoidDerivative
            
        if activationFunctionOutput == 'softMax':
            self.activationFunctionOutput = self.softMax
        

        self.create_biases_and_weights()
        

    #Define all new functions here:    

    def sigmoid(self, z):
        return 1./(1 + np.exp(-z))
    
    def sigmoidDerivative(self, z):
        return self.sigmoid(z)*(1. - self.sigmoid(z))
    
    def linear(self, z):
        return z
    
    def linearDerivative(self, z):
        return 1
    
    def softMax(self, z_o):
        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities
    
    def relu(self, x):
        return x * (x > 0)

    def reluDerivative(self, x):
        return 1. * (x > 0)


    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)*np.sqrt(2./self.n_features)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) 

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)*np.sqrt(2./self.n_hidden_neurons)
        self.output_bias = np.zeros(self.n_categories) 

    def feed_forward(self):
        # feed-forward for training
        self.z_h = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
        self.a_h = self.activationFunctionHidden(self.z_h) 

        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias

        self.a_o = self.activationFunctionOutput(self.z_o)
        
    def feed_forward_out(self, X):
        # feed-forward for output
        if X.ndim == 1:
            X = X.reshape(-1,1)
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = self.activationFunctionHidden(z_h)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        a_o = self.activationFunctionOutput(z_o)
        return a_o

        
        

    def backpropagation(self):

        error_output = self.a_o - self.Y_data
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.derivativeHidden(self.z_h)#* self.a_h * (1 - self.a_h) # changed

        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        if self.activationFunctionOutput != self.linear:
            return np.argmax(probabilities, axis=1)
        else:
            return probabilities
    

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]
                
                if self.X_data.ndim == 1:
                    self.X_data = self.X_data.reshape(-1,1)
                
                if self.Y_data.ndim == 1:
                    self.Y_data = self.Y_data.reshape(-1,1)
                    

                self.feed_forward()
                self.backpropagation()
                
    def trainMod(self):
 
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            
            indices = np.random.permutation(self.n_inputs)
            X = self.X_data_full[indices]
            y = self.Y_data_full[indices]
            for j in range(0, self.n_inputs, self.batch_size):
                
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                
                self.X_data = X[j:j+self.batch_size] 
                self.Y_data = y[j:j+self.batch_size]                 
                if self.X_data.ndim == 1:
                    self.X_data = self.X_data.reshape(-1,1)
                
                if self.Y_data.ndim == 1:
                    self.Y_data = self.Y_data.reshape(-1,1)
                    
                def learning_schedule(t, t0=5, t1=50):
                    return t0/(t+t1)
                
                self.eta = learning_schedule(i*self.iterations+j)

                self.feed_forward()
                self.backpropagation()
                
    def stochasticGradientDescentMod(self, nEpochs=50, batchSize=20,t0=5, t1 = 50):
        
        def learning_schedule(t):
            return t0/(t+t1)
        
        observations = len(self.yData)
        cost_history = np.zeros(nEpochs)
        n_batches = int(observations/batchSize)

        self.betaHat = np.random.random(self.features + 1) - .5
        
        costBest = 1e9
        for epoch in range(nEpochs):
            cost =0.0
            indices = np.random.permutation(observations)
            X = self.xData[indices]
            y = self.yData[indices]
            for i in range(0,observations,batchSize):
                X_i = X[i:i+batchSize]
                y_i = y[i:i+batchSize]
                
                X_i = np.c_[np.ones(len(X_i)),X_i]
                self.sigmoid(X_i @ self.betaHat)
                pHat = self.sigmoidOutput            
                gradients = -X_i.T @ (y_i - pHat)
                eta = learning_schedule(epoch*n_batches+i)
                self.betaHat -= eta*gradients
                cost += self.calculateCost(X_i,y_i)
            cost_history[epoch]  = cost
            if cost_history[epoch] < costBest:
                costBest = cost
                betaBest = self.betaHat
        return self.betaHat, cost_history, betaBest




class LogisticRegression:
    def __init__(self, xData, yData):
        self.xData, self.yData = xData, yData
        self.features = np.shape(self.xData)[1] 
        
    def createDesignMatrix(self):
        self.XHat = np.c_[np.ones(np.shape(self.xData)[0]), self.xData]
        
    def sigmoid(self, z):
        self.sigmoidOutput = 1./(1. + np.exp(-z))
        
    def gradientDescent(self, iterations = 1000, tolerance = 1e-8, eta=0.1):
        self.betaHat = np.random.uniform(low=-.05, high=.05, size=(self.features + 1))

        iteration = 1
        gradient = 10
        while (iteration < iterations and np.linalg.norm(gradient) > tolerance):
            self.sigmoid(self.XHat @ self.betaHat)
            pHat = self.sigmoidOutput            
            gradient = -self.XHat.T @ (self.yData - pHat)
            self.betaHat -= eta*gradient
            iteration += 1
        
        cost = self.calculateCost(self.XHat, self.yData)
        
        
    def stochasticGradientDescent(self,tolerance = 1e-8, n_epochs=50,t0=5, t1 = 50):
        """ From lecture notes"""
        
        def learning_schedule(t):
            return t0/(t+t1)

        
        self.betaHat = np.random.uniform(low=-.05, high=.05, size=(self.features + 1))
        
        m = len(self.yData)
        cost_history = np.zeros(n_epochs)

        for epoch in range(n_epochs):
            cost = 0
            costBest = 1e9
            for i in range(m):
                random_index = np.random.randint(m)
                xi = self.XHat[random_index:random_index+1]
                yi = self.yData[random_index:random_index+1]
                self.sigmoid(xi @ self.betaHat)
                pHat = self.sigmoidOutput            
                gradients = -xi.T @ (yi - pHat)
                eta = learning_schedule(epoch*m+i)
                self.betaHat -= eta*gradients
                cost += self.calculateCost(xi,yi)
            cost_history[epoch]  = cost
            if cost_history[epoch] < costBest:
                costBest = cost
                betaBest = self.betaHat
                
        return self.betaHat, cost_history, betaBest
            
                
                
    def stochasticGradientDescent2(self, nEpochs=50, batchSize=20,t0=5, t1 = 50):
        """  From lecture notes """
        
        def learning_schedule(t):
            return t0/(t+t1)
        
        observations = len(self.yData)
        cost_history = np.zeros(nEpochs)
        n_batches = int(observations/batchSize)

        self.betaHat = np.random.random(self.features + 1) - .5
        
        costBest = 1e9
        for epoch in range(nEpochs):
            cost =0.0
            indices = np.random.permutation(observations)
            X = self.xData[indices]
            y = self.yData[indices]
            for i in range(0,observations,batchSize):
                X_i = X[i:i+batchSize]
                y_i = y[i:i+batchSize]
                
                X_i = np.c_[np.ones(len(X_i)),X_i]
                self.sigmoid(X_i @ self.betaHat)
                pHat = self.sigmoidOutput            
                gradients = -X_i.T @ (y_i - pHat)
                eta = learning_schedule(epoch*n_batches+i)
                self.betaHat -= eta*gradients
                cost += self.calculateCost(X_i,y_i)
            cost_history[epoch]  = cost
            if cost_history[epoch] < costBest:
                costBest = cost
                betaBest = self.betaHat
        return self.betaHat, cost_history, betaBest
    
    def calculateCost(self, X, y):
        term1 = X @ self.betaHat
        term2 = np.log(1 + np.exp(X @ self.betaHat))
        ce = 0
        for i in range(len(y)):
            ce -= (y[i]*term1[i] - term2[i]) 
        return ce
                
            
    def predict(self, X):
        self.xData = X
        self.createDesignMatrix()
        self.sigmoid(self.XHat @ self.betaHat)
        return self.sigmoidOutput
        
        
    def predictHard(self, X, threshold = .5):
        prediction = self.predict(X)
        #self.predict(X)
        #prediction = self.sigmoidOutput
        return prediction >= threshold