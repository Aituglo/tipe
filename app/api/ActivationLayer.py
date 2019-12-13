from .Layer import Layer

# Récupère les classes de base de Layer
class ActivationLayer(Layer):

    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # Renvoie la sortie en passant par la fonction d'activation
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)

        return self.output

    # Renvoie l'erreur d'entrée dE/dX pour une erreur de sortie dE/dY donnée
    # Ici learning rate n'est pas utilisé car il n'y a pas de parametres qui peuvent apprendre
    def backward_propagation(self, output_error, learning_rate):
        # Renvoie dE/dY * f'(X)
        return output_error * self.activation_prime(self.input)