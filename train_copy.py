import theano
import numpy
import os
import random
import time
import cPickle

from ntm import *
from head import *
from head_neural import *
from controller import *
from controller_RNN import *
from theano import tensor as T
import tools
import task_data

import sys
sys.setrecursionlimit(10000)

if __name__ == '__main__':
	vector_size = 10
	memory_size = (32, 20)

	max_sequences = 500000
	one_round = 1
	max_seqlength = 20
	accu_score = -1
	best = 10000

	
	ntm = NTM(vector_size, Head_neural,1, MLPController, [50,vector_size], memory_size)
	for param in ntm.params:
		print param
		print param.get_value()

	#tmp = numpy.asarray([0.1,0.2,.3,.4,.5,0.1,0.2,.3,.4,.5])
	minval = 2.
	for i in xrange(max_sequences):
		seqlength = 20#numpy.random.randint(int(max_seqlength * (min(i,50000)/float(50000))**2) +1) + 1
		for j in xrange(one_round):
			input_sequence, output_sequence = task_data.copytask(vector_size, seqlength)
			#print ntm.train(input_sequence, output_sequence)
			#print ntm.test(input_sequence)
			score = ntm.train(input_sequence, output_sequence)
			if accu_score == -1:
				accu_score = score
			else:
				accu_score = 0.95*accu_score+0.05*score
			print 'iter: ',i*one_round+j,'error: ',score,'length:', seqlength
			'''
			if i > 10000 and accu_score < best:
				print 'saving model'
				savefile = open('copy/copy'+str(max_seqlength), 'wb')
				cPickle.dump(ntm,savefile,-1)
				savefile.close()
			'''
			if accu_score < best:
				print 'best'
				best = accu_score
			#print ntm.test(input_sequence, output_sequence)
			if score < minval:
				print input_sequence
				print output_sequence
				print ntm.predict(input_sequence)
				minval -= 0.1
				time.sleep(5)
		if (i*one_round+j) % 5000 == 0 and i > 0:
			for param in ntm.params:
				print param
				print param.get_value()
			input_sequence, output_sequence = task_data.copytask(vector_size,seqlength)
			print ntm.grads(input_sequence, output_sequence)
			print ntm.test()
			print input_sequence,output_sequence
			print ntm.predict(input_sequence)
			print 'best:',best
			time.sleep(10)

	print '-------finished-------'
	for param in ntm.params:
		print param
		print param.get_value()
	input_sequence, output_sequence = task_data.copytask(vector_size,2)
	print input_sequence,output_sequence
	print ntm.predict(input_sequence)
	