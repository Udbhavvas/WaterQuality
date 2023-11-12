import numpy as np

class SVM:

    def __init__(self, learning_rate=0.01, epochs=1000):
        # Constructor
        print("CONSTRUCTING")
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
        X = np.array(X)
        y = np.array(y)
        
        
        # Sets random weights ranging from -1 to +1
        num_samples, num_features = X.shape
        self.w = np.random.random_sample(num_features + 1)
        
        self.w = self.w * 2 - 1

        # Find hyperplane using stochastic gradient descent with hinge loss
        for _ in range(self.epochs):
            for idx, V in enumerate(X):
                # Use numpy.insert() to add the value at the front (position 0)
                V = np.insert(V, 0, [1])
                # 1 x1 x2
                #print(V)
                if y[idx] * (np.dot(self.w,np.transpose(V))) < 1:
                    #                     y[1] * V[(1 + p) x 1]
                    hinge_loss_gradient = y[idx] * V

                    self.w = self.w + np.multiply(self.learning_rate, hinge_loss_gradient)


        self.b = self.w[0]

    def accuracy(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        total = len(y_true)

        print(np.transpose(y_pred))
        print(y_true)

        accuracy = np.sum(y_true == y_pred) / total
        return accuracy

    def predict(self, X):
        # Include bias values somehow
        predictions = []
        for i in X:
            I = np.insert(i, 0, [1])
            scores = np.dot(I, np.transpose(self.w))
            predictions.append(np.sign(scores))

        return predictions

    def SVM_train_test_split(self, X, y, test_size, random_state):
        

        return 0