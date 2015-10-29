import theano
import numpy
import os

from theano import tensor as T
from controller import *
from memory import *
from head import *
import tools

class NTM(object):
	def __init__(self, vector_size, head_num, controller_sizes, memory_size, shift_width = 3, activation = T.tanh):
		self.lr = 0.001
		self.controller = MlpController(controller_sizes)
		self.input_w = theano.shared(value = tools.initial_weights((vector_size, controller_sizes[0])), name = 'input_w')
		self.input_b = theano.shared(value = tools.empty_floats(controller_sizes[0]), name = 'input_b')
		self.read_w = theano.shared(value = tools.initial_weights((vector_size, controller_sizes[0])), name = 'read_w')
		self.params = self.controller.params+[self.input_w, self.read_w, self.input_b]
		

		self.heads = []
		for i in xrange(head_num):
			self.heads.append(head(vector_size, memory_size[0], shift_width,i))
			self.params += self.heads[i].params

		#memory_init = tools.initial_weights(memory_size)
		memory_init = tools.initial_weights(memory_size)
		weight_init = numpy.asarray([1.0/memory_size[0]]*memory_size[0])
		print self.params

		#def weighting(weight, value):
		#	return weight*value

		def pred_t(rawinput_t, weight_tm1, memory_tm1):
			#memory_tm1 = self
			#predict the current output 
			input_t = T.dot(self.input_w, rawinput_t)
			read_m = T.dot(weight_tm1, memory_tm1)
			read_t = T.dot(self.read_w, read_m)
			controller_input = activation(input_t+read_t+self.input_b)
			result = self.controller.getY(controller_input)
			#result = read_m
			#emit the weights
			
			for head in self.heads:
				weight_t, erase, add, testin = head.emit_new_weight(result, weight_tm1, memory_tm1)
			#write to memory
			weight_tdim = weight_t.dimshuffle((0, 'x'))
			M_erased = memory_tm1*(1-weight_tdim*erase)
			memory_t = M_erased+weight_tdim*add
			
			return weight_t, memory_t, testin,result

		input = T.dmatrix()
		output = T.dmatrix()
		#tmp = T.dvector()
		#testinfo = self.controller.getY(input[1])
		testinfo = input.shape

		pred, _ = theano.scan(fn = pred_t, 
							sequences = input,
							outputs_info = [weight_init, memory_init, None,None ])

		#entropy = T.nnet.binary_crossentropy(1e-6+pred, 1e-6+output)
		
		costs = (pred[-1]-output) ** 2
		cost = T.sum(costs)

		grads = [T.grad(cost, param_i) for param_i in self.params]
		grads_clip = [T.clip(grad,-100,100) for grad in grads]
		updates = [(param_i, param_i-self.lr*grad_i) for param_i, grad_i in zip(self.params, grads)]
		
		self.predict = theano.function(inputs = [input], outputs = pred[-1])
		self.test = theano.function(inputs = [input], outputs = testinfo)
		#self.train = theano.function(inputs = [input,output], outputs = [input,output, costs, cost, pred[0],pred[1],pred[2],pred[-1]], updates = updates)
		self.train = theano.function(inputs = [input,output], outputs = cost, updates = updates)