import theano
import numpy
import os

def copytask(vector_size, length):
	sequence = numpy.asarray(
		numpy.random.uniform(
			low = -1.,
			high = 1.,
			size = (length, vector_size)
			),
			dtype = theano.config.floatX
		)
	
	input_sequence = numpy.zeros((2*length, vector_size), dtype = theano.config.floatX)
	output_sequence = numpy.zeros((2*length, vector_size), dtype = theano.config.floatX)
	input_sequence[0:length] = sequence
	output_sequence[length:2*length] = sequence
	return input_sequence, output_sequence