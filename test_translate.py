#-*- coding: utf-8 -*-
import theano
import numpy
import os
import random
import time
import cPickle

from ntm_translate import *
from train_translate import getsentence
from theano import tensor as T
import tools
import task_data
import json
import codecs

import sys
sys.setrecursionlimit(10000)

if __name__ == "__main__":
	ctest1filename = 'corpus/test.04.cn'
	cinput = open(ctest1filename, 'rb').read()
	test1outputs = [row.split(' ') for row in cinput.split('\n')]
	del test1outputs[-1]

	etest1filename = 'corpus/test.04.true.en'
	einput = open(etest1filename, 'rb').read()
	test1inputs = [row.split(' ')+['\s'] for row in einput.split('\n')]
	del test1inputs[-1]

	ctest2filename = 'corpus/test.05.cn'
	cinput = open(ctest1filename, 'rb').read()
	test2outputs = [row.split(' ') for row in cinput.split('\n')]
	del test2outputs[-1]

	etest2filename = 'corpus/test.05.true.en'
	einput = open(etest1filename, 'rb').read()
	test2inputs = [row.split(' ')+['\s'] for row in einput.split('\n')]
	del test2inputs[-1]

	ctest3filename = 'corpus/test.07.cn'
	cinput = open(ctest1filename, 'rb').read()
	test3outputs = [row.split(' ') for row in cinput.split('\n')]
	del test3outputs[-1]

	etest3filename = 'corpus/test.07.true.en'
	einput = open(etest1filename, 'rb').read()
	test3inputs = [row.split(' ')+['\s'] for row in einput.split('\n')]
	del test3inputs[-1]

	sourcedict = json.loads(open('corpus/sourcedict_1w', 'rb').read())
	targetdict = json.loads(open('corpus/targetdict_1w', 'rb').read())

	for input_set in [test1inputs, test2inputs, test3inputs]:
		for sentence in input_set:
			for i in range(0,len(sentence)):
				word = sentence[i]
				if word in sourcedict:
					sentence[i] = sourcedict.index(word)

	for output_set in [test1outputs, test2outputs, test3outputs]:
		for sentence in output_set:
			for i in range(0,len(sentence)):
				word = sentence[i]
				if word in targetdict:
					sentence[i] = targetdict.index(word)

	test1_input = []
	for input in test1inputs:
		test1_input.append(numpy.asarray(input+[0]*(len(input)-1)))
	test1_output = []
	for output in test1outputs:
		test1_output.append(output+[0])

	test2_input = []
	for input in test2inputs:
		test2_input.append(numpy.asarray(input+[0]*(len(input)-1)))
	test2_output = []
	for output in test2outputs:
		test2_output.append(output+[0])

	test3_input = []
	for input in test3inputs:
		test3_input.append(numpy.asarray(input+[0]*(len(input)-1)))
	test3_output = []
	for output in test3outputs:
		test3_output.append(output+[0])

	savefile = open('corpus/save/translation_1w_train.model', 'rb')
	ntm = cPickle.load(savefile)
	savefile.close()

	translation1 = ''
	for input, output in zip(test1_input,test1_output):
		translation1 += getsentence(ntm.predict(input)[0], targetdict)+'\n'

	output = codecs.open('corpus/translate.04.cn', 'wb', 'utf-8')
	output.write(translation1)
	output.close()

	translation2 = ''
	for input, output in zip(test2_input,test2_output):
		translation2 += getsentence(ntm.predict(input)[0], targetdict)+'\n'

	output = codecs.open('corpus/translate.05.cn', 'wb', 'utf-8')
	output.write(translation2)
	output.close()

	translation3 = ''
	for input, output in zip(test3_input,test3_output):
		translation3 += getsentence(ntm.predict(input)[0], targetdict)+'\n'

	output = codecs.open('corpus/translate.07.cn', 'wb', 'utf-8')
	output.write(translation3)
	output.close()
