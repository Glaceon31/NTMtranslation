import theano
import numpy
import os

from theano import tensor as T
import tools

class MLPController(object):
	def __init__(self, layer_sizes):
		#print layer_sizes
		self.layernum = len(layer_sizes)
		self.layer_sizes = layer_sizes
		w = []
		b = []
		self.w = []
		self.b = []
		#self.params = []
		self.params = []
		for i in xrange(self.layernum-1):
			#size = (layer_sizes[i], layer_sizes[i+1])
			w.append(tools.initial_weights(layer_sizes[i], layer_sizes[i+1]))
			b.append(0.*tools.initial_weights(layer_sizes[i+1]))
			self.w.append(theano.shared(value=w[i],name='Controller_w'+str(i), borrow=True))
			self.b.append(theano.shared(value=b[i],name='Controller_b'+str(i), borrow=True))
			self.params += [self.w[i], self.b[i]]

	def getY(self, input, activation=T.tanh):
		#if input.shape[0] != self.w[0].get_value().shape[0]:
		#	raise Exception('input length should be '+str(self.w[0].get_value().shape[0]))
		middle = input
		for i in xrange(self.layernum-1):
			middle = activation(T.dot(middle, self.w[i])+self.b[i])
			#middle = T.dot(middle, self.w[i])+self.b[i]
		fin_hid = middle
		#output = T.dot(middle, self.output_w)+self.output_b
		return fin_hid
		#return middle
