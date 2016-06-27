import theano
import numpy
import os
import math
import scipy

from theano import tensor as T
import tools
import theano.tensor.signal.conv

class Head_neural:
	def __init__(self, vector_size, memory_size, num):
		#context addressing
		key_w = tools.initial_weights(vector_size, memory_size[1])
		self.key_w = theano.shared(value = key_w, name = 'head%d_keyw' % num, borrow=True)
		key_b = 0.*tools.initial_weights(memory_size[1])
		self.key_b = theano.shared(value = key_b, name = 'head%d_keyb' % num, borrow=True)
		beta_w = tools.initial_weights(vector_size)
		self.beta_w = theano.shared(value = beta_w, name = 'head%d_betaw' % num, borrow=True)
		beta_b = numpy.asarray(0., dtype = theano.config.floatX)
		self.beta_b = theano.shared(value = beta_b, name = 'head%d_betab' % num, borrow=True)

		g_w = tools.initial_weights(vector_size)
		self.g_w = theano.shared(value = g_w, name = 'head%d_gw' % num, borrow=True)
		g_b = numpy.asarray(0., dtype = theano.config.floatX)
		self.g_b = theano.shared(value = g_b, name = 'head%d_gb' % num, borrow=True)
		location_w = tools.initial_weights(memory_size[0], memory_size[0])
		self.location_w = theano.shared(value = location_w, name = 'head%d_locationw' % num, borrow=True)
		location_b = 0.*tools.initial_weights(memory_size[0])
		self.location_b = theano.shared(value = location_b, name = 'head%d_locationb' % num, borrow = True)

		#erase and add
		erase_w = tools.initial_weights(vector_size,memory_size[1])
		self.erase_w = theano.shared(value = erase_w, name = 'head%d_erasew' % num, borrow=True)
		erase_b = 0.*tools.initial_weights(memory_size[1])
		self.erase_b = theano.shared(value = erase_b, name = 'head%d_eraseb' % num, borrow=True)
		add_w = tools.initial_weights(vector_size,memory_size[1])
		self.add_w = theano.shared(value = add_w, name = 'head%d_addw' % num, borrow=True)
		add_b = 0.*tools.initial_weights(memory_size[1])
		self.add_b = theano.shared(value = add_b, name = 'head%d_addb' % num, borrow=True)
		self.params = [self.key_w, self.key_b,self.beta_w, self.beta_b,self.g_w, self.g_b,\
						self.erase_w, self.erase_b, self.add_w, self.add_b,\
						self.location_w, self.location_b]

	def emit_new_weight(self, inp_h, weight_h, memory_h):
		#context addressing

		key = T.dot(inp_h, self.key_w)+self.key_b
		beta = T.nnet.softplus(T.dot(inp_h, self.beta_w )+self.beta_b)

		g = T.nnet.sigmoid(T.dot(inp_h, self.g_w)+self.g_b)

		#gamma = T.nnet.softplus(T.dot(inp_h,self.gamma_w)+self.gamma_b)
				
		weight_c = tools.vector_softmax(beta*tools.cos_sim(key, memory_h))
		#location addressing
		#interpolating
		
		weight_g = g*weight_c+ (1-g)*weight_h

		weight_location = T.tanh(T.dot(weight_g, self.location_w)+self.location_b)

		weight_new = weight_location
		
		#erase and add
		erase = T.nnet.sigmoid(T.dot(inp_h,self.erase_w)+self.erase_b)
		add = T.dot(inp_h,self.add_w)+self.add_b
		#if test:
		#	return key, beta,weight_c,g,weight_g,shift,weight_shift,gamma,weight_gamma,weight_new
		return weight_new, erase, add
		