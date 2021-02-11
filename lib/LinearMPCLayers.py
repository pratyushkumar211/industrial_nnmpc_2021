"""
Custom Neural Network layers for performing
experiments and comparisions with the MPC optimization problem.
Pratyush Kumar, pratyushkumar@ucsb.edu
"""
import sys
import time
import math
import numpy as np
import tensorflow as tf
from linearMPC import LinearMPCController
from python_utils import PickleTool
tf.keras.backend.set_floatx('float64')

class RegulatorLayerWithUprev(tf.keras.layers.Layer):
    """
    Linear MPC Regulator layer.
    Input to the layer: (x, xs*input_horizon, us*input_horizon)
    Output of the layer: (u*output_horizon).
    """

    def __init__(self, layer_dims, trainable=True, name=None):
        super(RegulatorLayerWithUprev, self).__init__(trainable, name)

        # Create a list which will contain all the internal dense layers.
        self._layers = []

        # Layers leading upto the final layer.
        for dim in layer_dims[:-1]:
            self._layers.append(tf.keras.layers.Dense(dim, activation='relu'))
        self._layers.append(tf.keras.layers.Dense(
            layer_dims[-1], use_bias=False))

    # The structured call function.
    # The final output is:
    # u = us + StrucNN(x, uprev, xs, us)
    # StrucNN := NN(x, uprev, xs, us) - NN(xs, us, xs, us)
    # Note that the above difference is same as the structured 
    # architecture presented in the paper.
    def call(self, inputs):

        # Get the inputs.
        [x, uprev, xs, us] = inputs

        # Propagate through the layers.
        layer_input = tf.concat((x, uprev, xs, us), axis=-1)
        for layer in self._layers[:-1]:
            layer_input = layer(layer_input)
        output1 = self._layers[-1](layer_input)

        # The output of the layer when the input is (xs, us, xs, us).
        layer_input = tf.concat((xs, us, xs, us), axis=-1)
        for layer in self._layers[:-1]:
            layer_input = layer(layer_input)
        output2 = self._layers[-1](layer_input)

        # Compute the final output.
        output2 = tf.negative(output2)
        final_output = tf.math.add(output1, output2)
        final_output = tf.math.add(us, final_output)
        return final_output

    def get_config(self):
        return super().get_config()

class RegulatorLayerWithoutUprev(tf.keras.layers.Layer):
    """
    Linear MPC Regulator layer.
    Input to the layer: (x, xs*input_horizon, us*input_horizon)
    Output of the layer: (u*output_horizon).
    """

    def __init__(self, layer_dims, trainable=True, name=None):
        super(RegulatorLayerWithoutUprev, self).__init__(trainable, name)

        # Create a list which will contain all the internal dense layers.
        self._layers = []

        # Layers leading upto the final layer.
        for dim in layer_dims[:-1]:
            self._layers.append(tf.keras.layers.Dense(dim, activation='relu'))
        self._layers.append(tf.keras.layers.Dense(
            layer_dims[-1], use_bias=False))

    # The structured call function.
    # The final output is:
    # u = us + StrucNN(x, xs, us)
    # StrucNN := NN(x, xs, us) - NN(xs, xs, us)
    # Note that the above difference is same as the structured
    # architecture presented in the paper.
    def call(self, inputs):

        # Get the inputs.
        [x, xs, us] = inputs        

        # Propagate through the layers.
        layer_input = tf.concat((x, xs, us), axis=-1)
        for layer in self._layers[:-1]:
            layer_input = layer(layer_input)
        output1 = self._layers[-1](layer_input)

        # The output of the layer when the input is (xs, xs, us).
        layer_input = tf.concat((xs, xs, us), axis=-1)
        for layer in self._layers[:-1]:
            layer_input = layer(layer_input)
        output2 = self._layers[-1](layer_input)

        # Compute the final output.
        output2 = tf.negative(output2)
        final_output = tf.math.add(output1, output2)
        final_output = tf.math.add(us, final_output)
        return final_output

    def get_config(self):
        return super().get_config()

class RegulatorModel(tf.keras.Model):
    """Custom regulator model, assumes 
        by default that the NN would take 
        uprev as an input."""
    def __init__(self, Nx, Nu, regulator_dims, nnwithuprev=True):
        
        inputs = [tf.keras.Input(name='x', shape=(Nx,)),
                  tf.keras.Input(name='xs', shape=(Nx,)),
                  tf.keras.Input(name='us', shape=(Nu,))]
        if nnwithuprev:
            inputs.insert(1, tf.keras.Input(name='uprev', shape=(Nu,)))
            regulator = RegulatorLayerWithUprev(layer_dims=regulator_dims[1:])
        else:
            regulator = RegulatorLayerWithoutUprev(layer_dims=
                                                   regulator_dims[1:])
        outputs = regulator(inputs)
        super().__init__(inputs=inputs, outputs=outputs)

class UnstdRegulatorLayer(tf.keras.layers.Layer):
    """
    Linear MPC Regulator layer.
    Input to the layer: (x, xs*input_horizon, us*input_horizon)
    Output of the layer: (u*output_horizon).
    """
    def __init__(self, layer_dims, trainable=True, name=None):
        super(UnstdRegulatorLayer, self).__init__(trainable, name)

        # Create a list which will contain all the internal dense layers.
        self._layers = []
        # Layers leading upto the final layer.
        for dim in layer_dims:
            self._layers.append(tf.keras.layers.Dense(dim, activation='relu'))

    # u = NN(x, xs, us)
    def call(self, inputs):
        # Propagate through the layers.
        output = tf.concat(inputs, axis=-1)
        for layer in self._layers:
            output = layer(output)
        return output

    def get_config(self):
        return super().get_config()

class UnstdRegulatorModel(tf.keras.Model):
    """Custom regulator model, assumes 
        by default that the NN would take 
        uprev as an input."""
    def __init__(self, Nx, Nu, regulator_dims, nnwithuprev=True):
        """ Unstructured model. """
        inputs = [tf.keras.Input(name='x', shape=(Nx,)),
                  tf.keras.Input(name='xs', shape=(Nx,)),
                  tf.keras.Input(name='us', shape=(Nu,))]
        if nnwithuprev:
            inputs.insert(1, tf.keras.Input(name='uprev', shape=(Nu,)))
        regulator = UnstdRegulatorLayer(layer_dims=regulator_dims[1:])
        outputs = regulator(inputs)
        super().__init__(inputs=inputs, outputs=outputs)