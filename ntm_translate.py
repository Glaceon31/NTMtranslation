import theano
import numpy
import os

from theano import tensor as T
from theano.ifelse import ifelse
from controller import *
from controller_RNN import *
from head import *
from head_neural import *
from theano.compile.debugmode import DebugMode
import tools

class NTM_translate(object):
	def __init__(self, vector_size, voc_size, head_type, head_num, controller_type, controller_sizes, memory_size,shift_width = 3, activation = T.tanh):
		self.controller = controller_type(controller_sizes)
		embedding = tools.initial_weights(voc_size[0]+1, vector_size)
		self.embedding = theano.shared(value = embedding, name = 'embedding', borrow=True)
		input_w = tools.initial_weights(vector_size, controller_sizes[0])
		self.input_w = theano.shared(value = input_w, name = 'input_w', borrow=True)
		input_b = 0.*tools.initial_weights(controller_sizes[0])
		self.input_b = theano.shared(value = input_b, name = 'input_b', borrow=True)
		read_w = tools.initial_weights(memory_size[1], controller_sizes[0])
		self.read_w = theano.shared(value = read_w, name = 'read_w', borrow=True)
		output_w = tools.initial_weights(controller_sizes[-1], voc_size[1])
		self.output_w = theano.shared(value=output_w,name='Controller_outw', borrow=True)
		output_b = 0.*tools.initial_weights(voc_size[1])
		self.output_b = theano.shared(value=output_b,name='Controller_outb', borrow=True)
		memory_init_p = 2*(numpy.random.rand(memory_size[0],memory_size[1])-0.5)
		weight_init_p = numpy.random.randn((memory_size[0]))
		self.memory_init = theano.shared(value = memory_init_p, name = 'memory_init', borrow=True)
		self.weight_init = theano.shared(value = weight_init_p, name = 'weight_init', borrow=True)
		self.params = self.controller.params+[self.embedding, self.input_w, self.read_w, self.input_b, self.weight_init,self.memory_init,self.output_w, self.output_b]

		memory_init = self.memory_init
		weight_init = tools.vector_softmax(self.weight_init)

		self.heads = []
		for i in xrange(head_num):
			if head_type == Head_neural:
				self.heads.append(head_type(controller_sizes[-1], memory_size,i))
			else:
				self.heads.append(head_type(controller_sizes[-1], memory_size, shift_width,i))
			self.params += self.heads[i].params
		print self.params

		def pred_t(input_voc_t, weight_tm1, memory_tm1):
			rawinput_t = self.embedding[input_voc_t]
			input_t = T.dot(rawinput_t,self.input_w)
			read_m = T.dot(weight_tm1, memory_tm1)
			read_t = T.dot(read_m,self.read_w)
			controller_input = activation(input_t+read_t+self.input_b)
			hid = self.controller.getY(controller_input)
			output = T.nnet.softmax(T.dot(hid, self.output_w)+self.output_b)
			result = T.switch(T.eq(input_voc_t, 0),T.argmax(output,axis=1), theano.shared(0))
			#test = controller_input
			
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
			

			return weight_t, memory_t, output,result


		input = T.lvector()
		output = T.lvector()

		pred, _ = theano.scan(fn = pred_t,
							sequences = [input],
							outputs_info = [weight_init, memory_init, None,None])

		p_output = -T.log(pred[-2])[output.shape[0]-1:]
		#output = output.reshape(output.shape[0],1)
		def cost_step(po, o,cost_tm1):
			cost = cost_tm1+po[0][o]
			return cost
		cost0 = theano.shared(0.)
		costs,_ = theano.scan(fn = cost_step,
			sequences = [p_output, output],
			outputs_info = [cost0]
			)

		l2 = T.sum(0)
		for param_i in self.params:
			l2 = l2+(param_i**2).sum()

		costs += 1e-4*l2

		grads = T.grad(costs[-1], self.params)
		grads_clip = [T.clip(grad,-100,100) for grad in grads]
		updates = tools.adadelta(self.params, grads_clip, 0.95, 1e-6)

		self.predict = theano.function(inputs = [input], outputs =[pred[-1]])
		self.train = theano.function(inputs= [input, output], outputs = costs[-1], updates = updates)
		self.test = theano.function(inputs= [input, output], outputs = costs[-1])
		self.getweight = theano.function(inputs = [input], outputs = [pred[0]])