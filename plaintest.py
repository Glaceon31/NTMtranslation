import theano
import numpy
import os

from theano import tensor as T

w = numpy.asarray([0.2,0.3,0.4], dtype = theano.config.floatX)
m = numpy.asarray([[.1, .3, .6,.5],[.4, .3, .6,.2],[.1, .7, .6,.0]])
e = numpy.asarray([.1, .2, .3, .4])

tw = T.vector()
tm = T.matrix()
te = T.vector()

twd = tw.dimshuffle((0, 'x'))
ted = te.dimshuffle(('x', 0))

result = tm*(1- (twd*ted))

fun = theano.function(inputs = [tw,tm,te], outputs= result)

print fun(w,m,e)