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
		self.memory = numpy.asarray(memory_size)

		input = T.dmatrix()
		testinfo = self.controller.getY(input[1])
		#testinfo = input[1].shape
		pred = input

		self.predict = theano.function(inputs = [input], outputs = pred)
		self.test = theano.function(inputs = [input], outputs = testinfo)