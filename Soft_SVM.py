import numpy as np

class SoftSVM:

    def __init__(self, learning_rate=0.01, epochs=1000):
        # Constructor
        print("CONSTRUCTING")
        
        self.w = None
        self.b = 0
        self.learning_rate = learning_rate
        self.epochs = epochs