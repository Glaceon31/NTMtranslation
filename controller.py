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
		self.layernum = len(layer_sizes)
		self.layer_sizes = layer_sizes
		output_w = tools.initial_weights(layer_sizes[-1], layer_sizes[-1])
		self.output_w = theano.shared(value=output_w,name='Controller_outw', borrow=True)
		output_b = 0.*tools.initial_weights(layer_sizes[-1])
		self.output_b = theano.shared(value=output_b,name='Controller_outb', borrow=True)
		w = []
		b = []
		self.w = []
		self.b = []
		#self.params = []
		self.params = [self.output_w, self.output_b]
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
		output = T.nnet.sigmoid(T.dot(middle, self.output_w)+self.output_b)
		#output = T.dot(middle, self.output_w)+self.output_b
		return output, fin_hid
		#return middle

