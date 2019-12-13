

class Layer:

    # Définie les données de base
    def __init__(self):
        self.input = None
        self.output = None

    # Va calculer la sortie Y par une couche avec une entrée X
    def forward_propagation(self, input):
        raise NotImplementedError

    # Va calculer dE/dX grace à dE/dY, et va changer les parametres si necessaire.
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError