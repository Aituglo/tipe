import numpy as np


# Fonction de perte MSE : Mean Squared Error
def mse(y_real, y_predict):
    return np.mean(np.power(y_real - y_predict, 2))

def mse_prime(y_real, y_predict):
    return 2*(y_predict - y_real)/y_real.size