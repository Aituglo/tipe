

class Network:

    def __init__(self):
        # Liste de couches
        self.layers = []
        # Fonction de perte
        self.loss = None
        # Sa dérivée
        self.loss_prime = None

    # Ajouter une couche à notre réseau
    def add(self, layer):
        self.layers.append(layer)

    # Ajoute une fonction de perte
    def add_loss(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # Prédit la sortie pour une entrée donnée
    def predict(self, input_data):
        size = len(input_data)
        result = []

        # Passe dans toutes les couches
        for i in range(size):
            # Forward Propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        
        return result

    # Entraine le réseau
    def train(self, x_train, y_train, n, learning_rate):
        size = len(x_train)
        log = []

        # Boucle d'entrainement
        for i in range(n):
            global_error = 0
            for j in range(size):
                # Forward Propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # Calcule la perte
                global_error += self.loss(y_train[j], output)

                # Backward Propagation, calcul de dE/dX
                error = self.loss_prime(y_train[j], output)
                # Passe les couches dans le sens inverse
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

                global_error /= size
                log.append("n : " + str(i+1) + "/" + str(n) + " error = " + str(global_error))

        return log