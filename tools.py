import theano
import numpy
import os

from theano import tensor as T
from itertools import izip

mini = 1e-5

def detect_nan(i, node, fn):
    for output in fn.outputs:
        if (not isinstance(output[0], numpy.random.RandomState) and
            numpy.isnan(output[0]).any()):
            print '*** NaN detected ***'
            theano.printing.debugprint(node)
            print 'Inputs : %s' % [input[0] for input in fn.inputs]
            print 'Outputs: %s' % [output[0] for output in fn.outputs]
            break
           
#theano.config.floatX = 'float32' 
def initial_weights(*argv):
	return numpy.asarray(
		numpy.random.uniform(
			low = -numpy.sqrt(6. / sum(argv)),
			high = numpy.sqrt(6. / sum(argv)),
			size = argv
			),
			dtype = theano.config.floatX
		)

def empty_floats(size):
	return numpy.zeros((size), dtype = theano.config.floatX)

def vec_dot(vec_a, vec_b):
	return T.sum(vec_a * vec_b)


def cos_similarity(vec_a, vec_b):
	#print T.dot(vec_a, vec_b)
	return vec_dot(vec_a, vec_b)/T.sqrt(vec_dot(vec_a, vec_a)*vec_dot(vec_b, vec_b))

def create_shared(array, dtype=theano.config.floatX, name=None):
	return theano.shared(
			value = numpy.asarray(
				array,
				dtype = dtype
			),
			name = name,
		)

def adadelta(parameters,gradients,rho,eps):
	# create variables to store intermediate updates
	gradients_sq = [ create_shared(numpy.zeros(p.get_value().shape)) for p in parameters ]
	deltas_sq = [ create_shared(numpy.zeros(p.get_value().shape)) for p in parameters ]
 
	# calculates the new "average" delta for the next iteration
	gradients_sq_new = [ rho*g_sq + (1-rho)*(g**2) for g_sq,g in izip(gradients_sq,gradients) ]
 
	# calculates the step in direction. The square root is an approximation to getting the RMS for the average value
	deltas = [ (T.sqrt(d_sq+eps)/T.sqrt(g_sq+eps))*grad for d_sq,g_sq,grad in izip(deltas_sq,gradients_sq_new,gradients) ]
 
	# calculates the new "average" deltas for the next step.
	deltas_sq_new = [ rho*d_sq + (1-rho)*(d**2) for d_sq,d in izip(deltas_sq,deltas) ]
 
	# Prepare it as a list f
	gradient_sq_updates = zip(gradients_sq,gradients_sq_new)
	deltas_sq_updates = zip(deltas_sq,deltas_sq_new)
	parameters_updates = [ (p,p - d) for p,d in izip(parameters,deltas) ]
	return gradient_sq_updates + deltas_sq_updates + parameters_updates

def clip(magnitude):
    def clipper(deltas):
        grads_norms = [ T.sqrt(T.sum(T.sqr(g))) for g in deltas ]
        return [ 
            T.switch(
                T.gt(n,magnitude),
                magnitude * (g/n),g
            ) for n,g in zip(grads_norms,deltas)
        ]
    return clipper

def vector_softmax(vec):
	return T.nnet.softmax(vec.reshape((1,vec.shape[0])))[0]

def cos_sim(k,M):

	k_unit = k / ( T.sqrt(T.sum(k**2)) + 1e-5 )
	k_unit = k_unit.dimshuffle(('x',0)) #T.patternbroadcast(k_unit.reshape((1,k_unit.shape[0])),(True,False))
	k_unit.name = "k_unit"
	M_lengths = T.sqrt(T.sum(M**2,axis=1)).dimshuffle((0,'x'))
	M_unit = M / ( M_lengths + 1e-5 )
	
	M_unit.name = "M_unit"
#	M_unit = Print("M_unit")(M_unit)
	return T.sum(k_unit * M_unit,axis=1)