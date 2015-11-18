import theano
import numpy
import os

from ntm import *
from head import *
from controller import *
from theano import tensor as T
import tools
import task_data

if __name__ == '__main__':
	vector_size = 3

	controller = MlpController([3,3])
	input_data = numpy.asarray([0., 1., 0.])

	
	input = T.dvector()
	test_controller = theano.function(inputs = [input], outputs = controller.getY(input))
	print test_controller(input_data)

	memory = numpy.asarray([[.2,.2,.2], [.1,-.6,.4],[-.8,.3,.9]])
	weight = numpy.asarray([.2,.3,.5])
	memory_size = (3,3)
	head = head(vector_size,memory_size,3,0)
	'''
	for param in head.params:
		print param
		print param.get_value()
	'''

	print '\n'
	'''
	test_head = theano.function(inputs=[input], outputs = head.emit_new_weight(input,weight,memory))
	for i in test_head(input_data):
		print i,'\n'
	'''

	#input_sequence, output_sequence = task_data.copytask(5,10)

	input_sequence = numpy.asarray([[1.,1.,0.],[0.,0.,1.],[0.,0.,0.]])
	output_sequence = numpy.asarray([[0.,0.,0.],[0.,0.,1.],[1.,1.,0.]])

	ntm = NTM(vector_size,1,[3,vector_size], memory_size)
	#input_sequence, output_sequence = task_data.copytask(vector_size, 2)

	for param in ntm.params:
		print param
		print param.get_value()
	print '\n'
	print input_sequence
	print 'pred\n'
	print ntm.predict(input_sequence)
	print 'grad\n'
	print ntm.grads(input_sequence,output_sequence)
	print 'train\n'
	print ntm.train(input_sequence,output_sequence)
	print 'newpred\n'
	print ntm.predict(input_sequence)
	print '\n'

	for param in ntm.params:
		print param
		print param.get_value()

	#print input_sequence
	#print output_sequence
	#ntm = NTM(3,1,[vector_size,3], (200, vector_size))
	#print ntm.params#.get_value()
	#params = ntm.params

	#input_sequence, output_sequence = task_data.copytask(vector_size,10)
	#print input_sequence#, output_sequence

	#print ntm.predict(input_sequence)
	#print ntm.test(input_sequence)
	#vec_a = numpy.asarray([1.,2.,3.]) 
	#vec_b = numpy.asarray([1.,2.,3.])
	#print tools.cos_similarity(vec_a, vec_b)