import theano
import numpy
import os

from theano import tensor as T
import tools

class Controller(object):
	pass

class ForwardController(Controller):
	pass

class MlpController(ForwardController):
	def __init__(self, layer_sizes):
		#print layer_sizes
		layernum = len(layer_sizes)
		self.layer_sizes = layer_sizes
		self.w = []
		self.b = []
		self.params = []
		for i in xrange(layernum-1):
			size = (layer_sizes[i], layer_sizes[i+1])
			self.w.append(theano.shared(value=tools.initial_weights(size),name='Controller_w'+str(i)))
			self.b.append(theano.shared(value=tools.empty_floats(layer_sizes[i+1]),name='Controller_b'+str(i)))
			self.params += [self.w[i], self.b[i]]

	def getY(self, input, activation=T.tanh):
		#if input.shape[0] != self.w[0].get_value().shape[0]:
		#	raise Exception('input length should be '+str(self.w[0].get_value().shape[0]))
		layernum = len(self.layer_sizes)
		middle = input
		for i in xrange(layernum-1):
			middle = activation(T.dot(middle, self.w[i])+self.b[i])
		output = middle
		return output

