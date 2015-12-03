#-*- coding: utf-8 -*-
import theano
import numpy
import os
import random
import time
import cPickle

from ntm_translate import *
from head import *
from head_neural import *
from controller import *
from controller_RNN import *
from theano import tensor as T
import tools
import task_data
import json

import sys
sys.setrecursionlimit(10000)

def getsentence(indexlist, dictionary, output0 = False):
	result = ''
	for i in indexlist:
		if type(i) == list:
			index = i[0]
		else:
			index = i
		#print i
		if index != 0:
			result += dictionary[index]+' '
		elif output0:
			result += '0 '
	return result

if __name__ == '__main__':
	sourcedict = ['\s','\UNK']
	targetdict = ['\s','\UNK']

	cfilename = 'corpus/toy.cn'
	#cfilename = 'Chinese.txt'
	cinput = open(cfilename, 'rb').read()
	outputs = [row.split(' ') for row in cinput.split('\n')]
	del outputs[-1]
	#print outputs
	efilename = 'corpus/toy.en'
	#efilename = 'English.txt'
	einput = open(efilename, 'rb').read()
	inputs = [row.split(' ')+['\s'] for row in einput.split('\n')]
	del inputs[-1]
	#print inputs

	for input_set in [inputs]:
		for sentence in input_set:
			for i in range(0,len(sentence)):
				word = sentence[i]
				if not word in sourcedict:
					sourcedict.append(word)
				sentence[i] = sourcedict.index(word)

	for output_set in [outputs]:
		for sentence in output_set:
			for i in range(0,len(sentence)):
				word = sentence[i]
				if not word in targetdict:
					targetdict.append(word)
				sentence[i] = targetdict.index(word)


	index_input = []
	for input in inputs:
		index_input.append(numpy.asarray(input+[0]*(len(input)-1)))
	index_output = []
	for output in outputs:
		index_output.append(output+[0])
	#train_input = numpy.asarray(train_input)
	#train_output = numpy.asarray(train_output)

	train_input = index_input[0:len(index_input)-2]
	train_output = index_output[0:len(index_output)-2]

	print len(sourcedict)
	print len(targetdict)
	print train_input
	print train_output

	ntm = NTM_translate(50, [len(sourcedict)+1, len(targetdict)+1], Head_neural,1, RNNController,[100,20], (256,50))

	print 'build NTM complete'

	print ntm.predict(train_input[1])
	count = 0
	best = 10000
	valve = 0.95
	

	for i in xrange(500):
		totalerror = 0
		for input,output in zip(train_input, train_output):
			cost = ntm.train(input, output)
			count += 1
			totalerror += cost
			print 'iter:', count, 'error:', cost
		print 'round:', i, 'totalerror:', totalerror
		if totalerror < best*valve:
			best = totalerror
			print 'saving model'
			savefile = open('corpus/model/toy', 'wb')
			cPickle.dump(ntm,savefile,-1)
			savefile.close()

	print 'best totalerror:',best
	print 'loading'
	loadfile = open('corpus/model/toy', 'rb')
	ntm_test = cPickle.load(loadfile)
	loadfile.close()

	for input,output in zip(index_input,index_output):
		print getsentence(input, sourcedict)
		print getsentence(ntm_test.predict(input)[0], targetdict)+'\n'
