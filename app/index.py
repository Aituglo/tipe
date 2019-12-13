from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np

from .api.Network import Network
from .api.FCLayer import FCLayer
from .api.ActivationLayer import ActivationLayer
from .api.Activations import tanh, tanh_prime
from .api.Loss import mse, mse_prime

from keras.datasets import mnist
from keras.utils import np_utils

app = Flask(__name__)

# Mnist Network
net_mnist = Network()
net_mnist.add(FCLayer(28*28, 100))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net_mnist.add(ActivationLayer(tanh, tanh_prime))
net_mnist.add(FCLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net_mnist.add(ActivationLayer(tanh, tanh_prime))
net_mnist.add(FCLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net_mnist.add(ActivationLayer(tanh, tanh_prime))
  
net_mnist.add_loss(mse, mse_prime)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
    

@app.route('/mnist/use', methods=['GET', 'POST'])
def mnist_use():
    
    # On recupere les datasets de mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if request.method == 'GET':

        # On met bien en forme nos données
        x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
        x_test = x_test.astype('float32')
        x_test /= 255

        # On met les resultats y sous la forme d'une matrice de taille 10
        # ex le nombre 3 sera [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        y_test = np_utils.to_categorical(y_test)

        start = int(request.args['start'])
        end = int(request.args['end'])

        out_test = net_mnist.predict(x_test[start:end])

        real = y_test[start:end]

        output = []

        for i in out_test:
            output.append(i.tolist())
       
        return jsonify(output=output ,real=real.tolist())

    elif request.method == 'POST':

        # dataset : On a 60000 exemples
        # On met bien en forme nos données
        x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
        x_train = x_train.astype('float32')
        x_train /= 255

        # On met les resultats y sous la forme d'une matrice de taille 10
        # ex le nombre 3 sera [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        y_train = np_utils.to_categorical(y_train)

        start = int(request.form['start'])
        end = int(request.form['end'])
        epochs = int(request.form['epochs'])
        learning = float(request.form['learning'])

        out_train = net_mnist.train(x_train[start:end], y_train[start:end], n=epochs, learning_rate=learning)

        return "Trained !"


@app.route('/mnist', methods=['GET'])
def mnist_view():
    return render_template('tests/mnist.html')


