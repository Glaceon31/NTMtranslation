import theano
import numpy
import os

from theano import tensor as T

def initialweights(size):
	return numpy.asarray(
		numpy.random.uniform(
			low = -numpy.sqrt(6. / sum(size)),
			high = numpy.sqrt(6. / sum(size)),
			size = size
			),
			dtype = theano.config.floatX
		)

def emptyfloat(size):
	return numpy.zeros((size), dtype = theano.config.floatX)