import theano
import numpy
import os

from ntm import *
from theano import tensor as T
import tools
import task_data

if __name__ == '__main__':
	vector_size = 3
	ntm = NTM(3,1,[vector_size,3], (200, vector_size))
	#print ntm.params#.get_value()
	params = ntm.params

	input_sequence, output_sequence = task_data.copytask(vector_size,10)
	#print input_sequence#, output_sequence

	print ntm.predict(input_sequence)
	#print ntm.test(input_sequence)
	vec_a = numpy.asarray([1.,2.,3.]) 
	vec_b = numpy.asarray([1.,2.,3.])
	#print tools.cos_similarity(vec_a, vec_b)