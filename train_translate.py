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
sys.setrecursionlimit(100000)

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

	cfilename = 'corpus/training.1w.cn'
	#cfilename = 'Chinese.txt'
	cinput = open(cfilename, 'rb').read()
	outputs = [row.split(' ') for row in cinput.split('\n')]
	del outputs[-1]
	#print outputs
	efilename = 'corpus/training.1w.en'
	#efilename = 'English.txt'
	einput = open(efilename, 'rb').read()
	inputs = [row.split(' ')+['\s'] for row in einput.split('\n')]
	del inputs[-1]
	#print inputs

	cvalidfilename = 'corpus/test.03.cn'
	cinput = open(cvalidfilename, 'rb').read()
	validoutputs = [row.split(' ') for row in cinput.split('\n')]
	del validoutputs[-1]

	evalidfilename = 'corpus/test.03.true.en'
	einput = open(evalidfilename, 'rb').read()
	validinputs = [row.split(' ')+['\s'] for row in einput.split('\n')]
	del validinputs[-1]

	ctestfilename = 'corpus/test.04.cn'
	cinput = open(ctestfilename, 'rb').read()
	testoutputs = [row.split(' ') for row in cinput.split('\n')]
	del testoutputs[-1]

	etestfilename = 'corpus/test.04.true.en'
	einput = open(etestfilename, 'rb').read()
	testinputs = [row.split(' ')+['\s'] for row in einput.split('\n')]
	del testinputs[-1]

	for input_set in [inputs, validinputs, testinputs]:
		for sentence in input_set:
			for i in range(0,len(sentence)):
				word = sentence[i]
				if not word in sourcedict:
					sourcedict.append(word)
				sentence[i] = sourcedict.index(word)

	for output_set in [outputs, validoutputs, testoutputs]:
		for sentence in output_set:
			for i in range(0,len(sentence)):
				word = sentence[i]
				if not word in targetdict:
					targetdict.append(word)
				sentence[i] = targetdict.index(word)

	#print sourcedict
	#print targetdict	



	index_input = []
	for input in inputs:
		index_input.append(numpy.asarray(input+[0]*(len(input)-1)))
	index_output = []
	for output in outputs:
		index_output.append(output+[0])
	#train_input = numpy.asarray(train_input)
	#train_output = numpy.asarray(train_output)

	train_input = index_input[0:len(index_input)]
	train_output = index_output[0:len(index_output)]

	index_input = []
	for input in validinputs:
		index_input.append(numpy.asarray(input+[0]*(len(input)-1)))
	index_output = []
	for output in validoutputs:
		index_output.append(output+[0])
	valid_input = index_input[0:len(index_input)]
	valid_output = index_output[0:len(index_output)]

	#print train_input
	#print train_output
	#print test_input
	#print test_output
	print 'train_corpus: ',len(train_input), 'sentences'
	print 'valid_corpus: ',len(valid_input), 'sentences'
	print 'source: ',len(sourcedict), 'words'
	print 'target: ',len(targetdict), 'words'
	sourcedictoutput = open('corpus/sourcedict_1w', 'wb')
	sourcedictoutput.write(json.dumps(sourcedict))
	sourcedictoutput.close()
	targetdictoutput = open('corpus/targetdict_1w', 'wb')
	targetdictoutput.write(json.dumps(targetdict))
	targetdictoutput.close()

	time.sleep(2)


	ntm = NTM_translate(50, [len(sourcedict)+1, len(targetdict)+1], Head_neural,2, RNNController, [100,50,20], (256,50))

	print 'build NTM complete'
	max_sequence = 500000
	endurance = 500000
	endurance_valve = 10000.0
	count = 0
	batch = 0
	best = 10000.0
	best_valid = 10000.0
	alpha = 0.95
	while count < endurance:
		batch += 1 
		total_error = 0
		#train 1 round
		for input,output in zip(train_input,train_output):
			count +=1
			#print getsentence(input,sourcedict,True)
			#print getsentence(output,targetdict,True)
			try:
				error = ntm.train(input,output)
				total_error += error
			except Exception, e:
				print 'error:', count
			if count % 1000 == 0:
				print 'batch:', batch,'iter:', count
			#print 'batch:', batch,'iter:', count, 'error:',ntm.train(input,output)
		#validate
		print 'batch:', batch,'iter:', count,'average error:', total_error/len(train_input)
		valid_error = 0.
		v_count = 0
		for input,output in zip(valid_input, valid_output):
			v_count += 1
			try:
				error = ntm.test(input,output)
				valid_error += error
			except Exception, e:
				print 'v_error:', v_count
		print 'batch:', batch,'iter:', count,'valid error:', valid_error/len(valid_input)
		if valid_error/len(valid_input) < best_valid:
			best_valid = valid_error/len(valid_input)
			print 'saving model valid'
			savefile = open('corpus/model/translation_1w_valid.model', 'wb')
			cPickle.dump(ntm,savefile,-1)
			savefile.close()
		if total_error/len(train_input) < alpha*endurance_valve:
			endurance_valve = total_error/len(train_input)
			endurance = min(max(endurance, 2*count),max_sequence)
			print 'endurance:', endurance
		if total_error/len(train_input) < best:
			best = total_error/len(train_input)
			print 'saving model'
			savefile = open('corpus/model/translation_1w_train.model', 'wb')
			cPickle.dump(ntm,savefile,-1)
			savefile.close()
			#save model
		#time.sleep(0.5)

	#translate
	translation = ''
	for input,output in zip(train_input,train_output):
		translation += getsentence(ntm.predict(input)[0], targetdict)+'\n'
		#print 'source:', getsentence(input,sourcedict, True)
		#print ntm.predict(input)[0]
		#print 'target:', getsentence(ntm.predict(input)[0], targetdict, True)
		#print ''

	output = open('corpus/translate.cn', 'wb')
	output.write(translation)
	output.close()


			



