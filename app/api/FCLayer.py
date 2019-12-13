from .Layer import Layer
import numpy as np 

# Récupère les classes de base de Layer
class FCLayer(Layer):

    # Il nous faut de base le nombre de neurone d'entrée et de sortie
    def __init__(self, input_size, output_size):
        # Donne une valeur par défault au poid
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # Renvoie la sortie pour une entrée
    def forward_propagation(self, input_data):
        self.input = input_data
        # Calcule Y = X*W + B
        self.output = self.bias + np.dot(self.input, self.weights)

        return self.output

    def backward_propagation(self, output_error, learning_rate):
        # Calcule dE/dX en faisant dE/dY * la transposé de W
        input_error = np.dot(output_error, self.weights.T)
        # Calcule dE/dW en faisant la transposé de X * dE/dY
        weights_error = np.dot(self.input.T, output_error)
        # On a dE/dB = dE/dY
        bias_error = output_error

        # Met à jour les paramètres
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error

        return input_error
        