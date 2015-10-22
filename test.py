import theano
import numpy
import os

from ntm import *
from theano import tensor as T
import task_data

if __name__ == '__main__':
	ntm = NTM([3,3,2], (200, 3))
	print ntm.controller.w[0].get_value().shape
	params = ntm.params

	input_sequence, output_sequence = task_data.copytask(3,10)
	#print input_sequence, output_sequence

	print ntm.predict(input_sequence)
	print ntm.test(input_sequence)