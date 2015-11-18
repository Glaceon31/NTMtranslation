import theano
import numpy
import os
import math
import scipy

from theano import tensor as T
import tools
import theano.tensor.signal.conv

class head:
	def __init__(self, vector_size, memory_size, shift_width, num):
		#context addressing
		key_w = tools.initial_weights(vector_size, memory_size[1])
		self.key_w = theano.shared(value = key_w, name = 'head%d_keyw' % num, borrow=True)
		key_b = 0.*tools.initial_weights(memory_size[1])
		self.key_b = theano.shared(value = key_b, name = 'head%d_keyb' % num, borrow=True)
		beta_w = tools.initial_weights(vector_size)
		self.beta_w = theano.shared(value = beta_w, name = 'head%d_betaw' % num, borrow=True)
		beta_b = numpy.asarray(0., dtype = theano.config.floatX)
		self.beta_b = theano.shared(value = beta_b, name = 'head%d_betab' % num, borrow=True)

		#location addressing
		#interpolation
		#self.shift_conv = scipy.linalg.circulant(numpy.arange(memory_size[0])).T[numpy.arange(-(shift_width//2),(shift_width//2)+1)][::-1]
		self.shift_width = shift_width
		g_w = tools.initial_weights(vector_size)
		self.g_w = theano.shared(value = g_w, name = 'head%d_gw' % num, borrow=True)
		g_b = numpy.asarray(0., dtype = theano.config.floatX)
		self.g_b = theano.shared(value = g_b, name = 'head%d_gb' % num, borrow=True)
		shift_w = tools.initial_weights(vector_size,shift_width)
		self.shift_w = theano.shared(value = shift_w, name = 'head%d_shiftw' % num, borrow=True)
		shift_b = 0.*tools.initial_weights(shift_width)
		self.shift_b = theano.shared(value = shift_b, name = 'head%d_shiftb' % num, borrow=True)
		gamma_w = tools.initial_weights(vector_size)
		self.gamma_w = theano.shared(value = gamma_w, name = 'head%d_gammaw' % num, borrow=True)
		gamma_b = numpy.asarray(0., dtype = theano.config.floatX)
		self.gamma_b = theano.shared(value = gamma_b, name = 'head%d_gammab' % num, borrow=True)

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
						self.erase_w, self.erase_b, self.add_w, self.add_b]
		#self.params = [self.erase_w, self.erase_b, self.add_w, self.add_b]
	def emit_new_weight(self, inp_h, weight_h, memory_h):
		#context addressing
		
		
		key = T.dot(inp_h, self.key_w)+self.key_b
		beta = T.nnet.softplus(T.dot(inp_h, self.beta_w )+self.beta_b)

		g = T.nnet.sigmoid(T.dot(inp_h, self.g_w)+self.g_b)

		#shift = tools.vector_softmax(T.dot(inp_h,self.shift_w)+self.shift_b)
		#shift = shift.dimshuffle((0,'x'))

		#gamma = T.nnet.softplus(T.dot(inp_h,self.gamma_w)+self.gamma_b)+1.
		

		'''
		key_normal = key/(T.sqrt(T.sum(key**2)) + tools.mini)
		memory_mod = T.sqrt(T.sum(memory**2, axis = 1).dimshuffle((0,'x')))
		memory_normal = memory/(memory_mod  + tools.mini)	

		weight_c = T.exp(beta*T.dot(memory_normal, key_normal))
		weight_cnormal = weight_c/T.sum(weight_c  + tools.mini)
		'''

		
		weight_c = tools.vector_softmax(beta*tools.cos_sim(key, memory_h))
		#location addressing
		#interpolating
		
		weight_g = g*weight_c+ (1-g)*weight_h
		#weight_conv = theano.tensor.signal.conv(weight_g.reshape(memory.shape[0],1),
		#				shift_normal.reshape(self.shift_width, 1))

		#code from shaw

		#shift_normal = shift/T.sum(shift)
		
		#weight_shift = T.sum(shift*weight_g[self.shift_conv], axis = 0)
		#sharpening
		
		#weight_gamma = weight_shift ** gamma
		#weight_gamma = weight_g
		
		#weight_new = weight_gamma/T.sum(weight_gamma)
		weight_new = weight_g

		#erase and add
		erase = T.nnet.sigmoid(T.dot(inp_h,self.erase_w)+self.erase_b)
		add = T.dot(inp_h,self.add_w)+self.add_b
		#if test:
		#	return key, beta,weight_c,g,weight_g,shift,weight_shift,gamma,weight_gamma,weight_new
		return weight_new, erase, add