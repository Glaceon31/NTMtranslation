import theano
import numpy
import os

from theano import tensor as T

mini = 1e-5

def initial_weights(size):
	return numpy.asarray(
		numpy.random.uniform(
			low = -numpy.sqrt(6. / sum(size)),
			high = numpy.sqrt(6. / sum(size)),
			size = size
			),
			dtype = theano.config.floatX
		)

def empty_floats(size):
	return numpy.zeros((size), dtype = theano.config.floatX)

def vec_dot(vec_a, vec_b):
	return T.sum(vec_a * vec_b)


def cos_similarity(vec_a, vec_b):
	#print T.dot(vec_a, vec_b)
	return vec_dot(vec_a, vec_b)/T.sqrt(vec_dot(vec_a, vec_a)*vec_dot(vec_b, vec_b))
