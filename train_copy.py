import theano
import numpy
import os
import random
import time

from ntm import *
from theano import tensor as T
import tools
import task_data

if __name__ == '__main__':
	vector_size = 8
	memory_size = (128, 20)
	ntm = NTM(vector_size,1,[100,vector_size], memory_size)
	for param in ntm.params:
		print param
		print param.get_value()

	max_sequences = 500000
	one_round = 1
	max_seqlength = 2

	#tmp = numpy.asarray([0.1,0.2,.3,.4,.5,0.1,0.2,.3,.4,.5])
	minval = 2.
	for i in xrange(max_sequences):
		seqlength = 2#numpy.random.randint(int(max_seqlength * (min(i,50000)/float(50000))**2) +1) + 1
		for j in xrange(one_round):
			input_sequence, output_sequence = task_data.copytask(vector_size, seqlength)
			#print ntm.train(input_sequence, output_sequence)
			score = ntm.train(input_sequence, output_sequence)
			print 'iter: ',i*one_round+j,'error: ',score,'length:', seqlength
			#print ntm.test(input_sequence, output_sequence)
			if score < minval:
				print input_sequence
				print output_sequence
				print ntm.predict(input_sequence)
				minval -= 0.1
				time.sleep(5)
		if (i*one_round+j) % 5000 == 0:
			for param in ntm.params:
				print param
				print param.get_value()
			input_sequence, output_sequence = task_data.copytask(vector_size,2)
			print input_sequence,output_sequence
			print ntm.predict(input_sequence)
			time.sleep(10)

	print '-------finished-------'
	for param in ntm.params:
		print param
		print param.get_value()
	input_sequence, output_sequence = task_data.copytask(vector_size,2)
	print input_sequence,output_sequence
	print ntm.predict(input_sequence)