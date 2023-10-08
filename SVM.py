import numpy as np

class SVM:

    def __init__(self, learning_rate=0.05, epochs=1000):
        # Constructor

        self.w = None
        self.b = 0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def edit():
        # Edits hyperparameters
        
        pass
    
    def train(self, X, y):
        # Calibrates model on input data
        # X is an NxM matrix, where N is the number of samples, and M is the number of features
        # y is an N dimensional array of labels for the samples in X
        # Returns nothing

        # Makes sure inputs are numpy compatible
        X = np.matrix(X)
        y = np.array(y)
        
        
        # Sets random weights ranging from -1 to +1
        num_samples, num_features = X.shape
        self.w = np.random.random_sample(num_features)
        
        self.w = self.w * 2 - 1

        # Find hyperplane using stochastic gradient descent with hinge loss
        for _ in range(self.epochs):
            for idx, V in enumerate(X):
                if y[idx] * (np.dot(self.w,np.transpose(V[0]))) < 1:
                    hinge_loss_gradient = y[idx] * V[0]
                    #print(V)
                    self.w = self.w - self.learning_rate * (hinge_loss_gradient)
                    self.b = self.b + y[idx]
        
        

    def accuracy(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        total = len(y_true)

        accuracy = np.sum(y_true == np.transpose(y_pred)) / total
        return accuracy

    def predict(self, X):
        # Include bias values somehow
        predictions = []
        for i in X:
            scores = np.dot(i, np.transpose(self.w)) + self.b
            predictions.append(np.sign(scores))

        return predictions

    def SVM_train_test_split(self, X, y, test_size, random_state):
        
        # if random_state is not None:
        #     np.random.seed(random_state)

        # X.shape[0]
        return 0
            
