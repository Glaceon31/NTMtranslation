import theano
import numpy
import os

from theano import tensor as T
from controller import *
from memory import *
import tools

class NTM(object):
	def __init__(self, controller_sizes, memory_size):
		self.controller = MlpController(controller_sizes)
		self.params = self.controller.params
		self.memory = numpy.zeros(memory_size)

		def pred_one(input_one):
			return self.controller.getY(input_one)

		input = T.dmatrix()
		#testinfo = self.controller.getY(input[1])
		testinfo = input.shape
		pred, _ = theano.scan(fn = pred_one, 
								sequences = input)
		'''
		for i in xrange(input.shape[0]):
			pred[i] = self.controller.getY(input[i])
		'''


		self.predict = theano.function(inputs = [input], outputs = pred)
		self.test = theano.function(inputs = [input], outputs = testinfo)