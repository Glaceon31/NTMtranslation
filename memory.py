import theano
import numpy
import os

from theano import tensor as T

class memory(object):
	def __init__(self, size):
		self.memory