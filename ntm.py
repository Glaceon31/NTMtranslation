import theano
import numpy
import os

from theano import tensor as T
from theano.ifelse import ifelse
from controller import *
from head import *
from theano.compile.debugmode import DebugMode
import tools

class NTM(object):
	def __init__(self, vector_size, head_num, controller_sizes, memory_size, shift_width = 3, activation = T.tanh):
		self.lr = 0.01
		self.controller = MlpController(controller_sizes)
		input_w = tools.initial_weights(vector_size, controller_sizes[0])
		self.input_w = theano.shared(value = input_w, name = 'input_w', borrow=True)
		input_b = 0.*tools.initial_weights(controller_sizes[0])
		self.input_b = theano.shared(value = input_b, name = 'input_b', borrow=True)
		read_w = tools.initial_weights(memory_size[1], controller_sizes[0])
		self.read_w = theano.shared(value = read_w, name = 'read_w', borrow=True)
		memory_init_p = 2*(numpy.random.rand(memory_size[0],memory_size[1])-0.5)
		weight_init_p = numpy.random.randn((memory_size[0]))
		self.memory_init = theano.shared(value = memory_init_p, name = 'memory_init', borrow=True)
		self.weight_init = theano.shared(value = weight_init_p, name = 'weight_init', borrow=True)
		self.params = self.controller.params+[self.input_w, self.read_w, self.input_b, self.weight_init,self.memory_init]

		self.heads = []
		for i in xrange(head_num):
			self.heads.append(head(vector_size, memory_size, shift_width,i))
			self.params += self.heads[i].params

		#memory_init = tools.initial_weights(memory_size)
		memory_init = self.memory_init
		#weight_init_s = T.nnet.sigmoid(self.weight_init)
		weight_init = tools.vector_softmax(self.weight_init)
		print self.params

		#def weighting(weight, value):
		#	return weight*value

		def pred_t(rawinput_t, weight_tm1, memory_tm1):
			#memory_tm1 = self
			#predict the current output 
			input_t = T.dot(rawinput_t,self.input_w)
			read_m = T.dot(weight_tm1, memory_tm1)
			read_t = T.dot(read_m,self.read_w)
			controller_input = activation(input_t+read_t+self.input_b)
			#zero_vec = theano.shared(value=numpy.zeros((vector_size,)))
			#mask = T.nonzero(T.eq(rawinput_t,0))
			output, hid = self.controller.getY(controller_input)
			#result = T.switch(T.eq(zero_vec,rawinput_t),output,theano.shared(0))
			result = output
			#testing = T.switch(T.eq(zero_vec,rawinput_t),theano.shared(1),theano.shared(0))
			#result = theano.shared(value=numpy.zeros((vector_size,)))
			
			#result = read_m
			#emit the weights
			
			memory_inter = memory_tm1
			weight_inter = weight_tm1
			for head in self.heads:
				weight_inter, erase, add= head.emit_new_weight(hid, weight_inter, memory_inter)
				#write to memory
				weight_tdim = weight_inter.dimshuffle((0, 'x'))
				erase_dim = erase.dimshuffle(('x', 0))
				add_dim = add.dimshuffle(('x', 0))
				M_erased = memory_inter*(1-(weight_tdim*erase_dim))
				memory_inter = M_erased+(weight_tdim*add_dim)

			#testing = weight_tm1
			#testing2 = rawinput_t
			memory_t = memory_inter
			weight_t = weight_inter

			return weight_t, memory_t, result

		input = T.matrix()
		output = T.matrix()
		#tmp = T.dvector()
		#testinfo = self.controller.getY(input[1])
		#testinfo = input.shape

		pred, _ = theano.scan(fn = pred_t, 
							sequences = [input],
							outputs_info = [weight_init, memory_init,None ])

		
		entropy = T.sum(T.nnet.binary_crossentropy(5e-6+(1-1e-5)*pred[-1], output),axis = 1)
		
		
		#costs = (pred[-1]-output) ** 2
		#cost_sq = T.sum(costs)


		l2 = T.sum(0)
		for param_i in self.params:
			l2 = l2+(param_i**2).sum()
		cost = T.sum(entropy) +1e-3*l2


		grads = [T.grad(cost, param_i) for param_i in self.params]
		grads_clip = [T.clip(grad,-100,100) for grad in grads]
		#params_up = [param_i for param_i, grad_i in zip(self.params, grads_clip)]
		#new_value = [param_i-self.lr*grad_i for param_i, grad_i in zip(self.params, grads_clip)]
		#SGD
		#updates = [(param_i, param_i-self.lr*grad_i) for param_i, grad_i in zip(self.params, grads_clip)]
		#updates = zip(params_up, new_value)
		
		#adadelta
		updates = tools.adadelta(self.params, grads_clip, 0.95, 1e-6)
		#updates = tools.adadelta_another(self.params,grads_clip)

		self.predict = theano.function(inputs = [input], outputs = [weight_init]+[memory_init]+pred)
		self.grads = theano.function(inputs = [input, output], outputs = grads)
		#self.train = theano.function(inputs = [input,output], outputs = [input,output, costs, cost, pred[0],pred[2],pred[-1],grads[5],grads_clip[5], grads[6]], updates = updates)
		self.train = theano.function(inputs = [input,output], outputs = cost, updates = updates)#,mode=theano.compile.MonitorMode(post_func=tools.detect_nan))