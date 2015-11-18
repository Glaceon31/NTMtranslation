import theano
import numpy
import os

def copytask(vector_size, length):
	'''
	sequence = numpy.asarray(
		numpy.random.uniform(
			low = -1.,
			high = 1.,
			size = (length, vector_size-1)
			),
			dtype = theano.config.floatX
		)
	'''
	sequence = numpy.random.binomial(1,0.5,(length, vector_size-1)).astype(numpy.uint8)
	#print sequence
	input_sequence = numpy.zeros((2*length+1, vector_size), dtype = numpy.float32)
	output_sequence = numpy.zeros((2*length+1, vector_size), dtype = numpy.float32)
	input_sequence[:length,:-1] = sequence
	input_sequence[length, -1] = 1
	output_sequence[length+1:,:-1] = sequence
	return input_sequence, output_sequence