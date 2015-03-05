from __future__ import division
import numpy as np
import sys
from functools import partial
from sklearn import metrics
from sklearn.feature_extraction.text import *
from sklearn.preprocessing import *
from theano.ifelse import ifelse
import theano
import theano.tensor as T
import pyprind

TRAIN_FILE='data/en/qa2_two-supporting-facts_train.txt' #sys.argv[1]
TEST_FILE='data/en/qa2_two-supporting-facts_test.txt' #sys.argv[2]
print "train: ", TRAIN_FILE
print "test: ", TEST_FILE

D = 50
gamma = 0.1

def zeros(shape, dtype=np.float32):
    return np.zeros(shape, dtype)

def O_t(xs, L, s):
    t = 0
    for i in xrange(1, len(L)):
        if s(xs, i, t, L) > 0:
            t = i
    return t

def get_train(U_Ot, U_R, lenW):
    def phi_x1(x_t, L):
        return T.concatenate([L[x_t].reshape((-1,)), zeros((2*lenW,)), zeros((3,))], axis=0)
    def phi_x2(x_t, L):
        return T.concatenate([zeros((lenW,)), L[x_t].reshape((-1,)), zeros((lenW,)), zeros((3,))], axis=0)
    def phi_y(x_t, L):
        return T.concatenate([zeros((2*lenW,)), L[x_t].reshape((-1,)), zeros((3,))], axis=0)
    def phi_t(x_t, y_t, yp_t, L):
        return T.concatenate([zeros(3*lenW,), T.stack(T.switch(T.lt(x_t,y_t), 1, 0), T.switch(T.lt(x_t,yp_t), 1, 0), T.switch(T.lt(y_t,yp_t), 1, 0))], axis=0)
    def s_Ot(xs, y_t, yp_t, L):
        result, updates = theano.scan(
            lambda x_t, t: T.dot(T.dot(T.switch(T.eq(t, 0), phi_x1(x_t, L).reshape((1,-1)), phi_x2(x_t, L).reshape((1,-1))), U_Ot.T),
                                 T.dot(U_Ot, (phi_y(y_t, L) - phi_y(yp_t, L) + phi_t(x_t, y_t, yp_t, L)))),
            sequences=[xs, T.arange(T.shape(xs)[0])])
        return result.sum()
    def sR(xs, y_t, L, V):
        result, updates = theano.scan(
            lambda x_t, t: T.dot(T.dot(T.switch(T.eq(t, 0), phi_x1(x_t, L).reshape((1,-1)), phi_x2(x_t, L).reshape((1,-1))), U_R.T),
                                 T.dot(U_R, phi_y(y_t, V))),
            sequences=[xs, T.arange(T.shape(xs)[0])])
        return result.sum()

    x_t = T.iscalar('x_t')
    m_o1 = T.iscalar('m_o1')
    m_o2 = T.iscalar('m_o2')
    r_t = T.iscalar('r_t')
    gamma = T.scalar('gamma')
    f1_t = T.iscalar('f1_t')
    f2_t = T.iscalar('f2_t')
    L = T.fmatrix('L') # list of messages
    V = T.fmatrix('V') # vocab
#    r_args = T.switch(T.eq(m_o2, -1.0), T.stack(x_t, m_o1), T.stack(x_t, m_o1, m_o2))
    r_args = T.stack(x_t, m_o1, m_o2)

    cost1, u1 = theano.scan(
        lambda f_bar, t: T.largest(gamma - s_Ot(T.stack(x_t), f1_t, t, L), 0),
        sequences=[L, T.arange(T.shape(L)[0])])

    cost2, u2 = theano.scan(
        lambda f_bar, t: T.largest(gamma + s_Ot(T.stack(x_t), t, f1_t, L), 0),
        sequences=[L, T.arange(T.shape(L)[0])])

    cost3, u3 = theano.scan(
        lambda f_bar, t: T.largest(gamma - s_Ot(T.stack(x_t, m_o1), f2_t, t, L), 0),
        sequences=[L, T.arange(T.shape(L)[0])])

    cost4, u4 = theano.scan(
        lambda f_bar, t: T.largest(gamma + s_Ot(T.stack(x_t, m_o1), t, f2_t, L), 0),
        sequences=[L, T.arange(T.shape(L)[0])])

    cost5, u5 = theano.scan(
        lambda r_bar, t: T.largest(gamma - sR(r_args, r_t, L, V) + sR(r_args, t, L, V), 0),
        sequences=[V, T.arange(T.shape(V)[0])])

    cost = cost1.sum()  + cost2.sum() + cost3.sum() + cost4.sum() + cost5.sum() - 5*gamma
    g_uo, g_ur = T.grad(cost, [U_Ot, U_R])

    train = theano.function(
        inputs=[x_t, f1_t, f2_t, r_t, m_o1, m_o2, gamma, L, V],
        outputs=[cost],
        updates=[(U_Ot, U_Ot-0.01*g_uo), (U_R, U_R-0.01*g_ur)])
    return train

def get_lines(fname):
    lines = []
    for line in open(fname):
        id = int(line[0:line.find(' ')])
        line = line.strip()
        line = line[line.find(' ')+1:]        
        if line.find('?') == -1:
            lines.append({'type':'s', 'text': line})
        else:
            idx = line.find('?')
            tmp = line[idx+1:].split('\t')
            lines.append({'id':id, 'type':'q', 'text': line[:idx], 'answer': tmp[1].strip(), 'refs': [int(x) for x in tmp[2:][0].split(' ')]})
    return np.array(lines)

def do_train(fname, vectorizer):    
    def phi_x1(x_t, L):
        return np.concatenate([L[x_t].reshape((-1,)), zeros((2*lenW,)), zeros((3,))], axis=0)
    def phi_x2(x_t, L):
        return np.concatenate([zeros((lenW,)), L[x_t].reshape((-1,)), zeros((lenW,)), zeros((3,))], axis=0)
    def phi_y(y_t, L):
        return np.concatenate([zeros((2*lenW,)), L[y_t].reshape((-1,)), zeros((3,))], axis=0)
    def phi_t(x_t, y_t, yp_t, L):
        return np.concatenate([zeros(3*lenW,), [1 if x_t < y_t else 0, 1 if x_t < yp_t else 0, 1 if y_t < yp_t else 0]])
    def sR(xs, y_t, L, V):
        def s(x, y, U):
            return np.dot(np.dot(x.reshape((1,-1)), U.T), np.dot(U, y))
        result = 0
        for i,x_t in enumerate(xs):
            result += s(phi_x1(x_t, L) if i == 0 else phi_x2(x_t, L), phi_y(y_t, V), U_R.get_value())
        return result
    def s_Ot(xs, y_t, yp_t, L):
        result = 0
        for i,x_t in enumerate(xs):
            x = phi_x1(x_t, L) if i == 0 else phi_x2(x_t, L)
            y = phi_y(y_t, L)
            yp = phi_y(yp_t, L)
            U = U_Ot.get_value()
            result += np.dot(np.dot(x.reshape((1,-1)), U.T), np.dot(U, y - yp + phi_t(x_t, y_t, yp_t, L)))
        return result

    lines = get_lines(fname)
    L = vectorizer.fit_transform([x['text'] for x in lines]).toarray().astype(np.float32)
    lenW = len(vectorizer.vocabulary_)
    H = {}
    for i,v in enumerate(vectorizer.vocabulary_):
        H[v] = i
    V = vectorizer.transform([v for v in vectorizer.vocabulary_]).toarray().astype(np.float32)

    W = 3*lenW + 3
    U_Ot = theano.shared(np.random.randn(D, W).astype(np.float32))
    U_R = theano.shared(np.random.randn(D, W).astype(np.float32))
    train = get_train(U_Ot, U_R, lenW)

    for epoch in range(10):
        total_err = 0
        print "*" * 80
        print "epoch: ", epoch
        for i,line in enumerate(lines):
            if i % 100 == 0:
                print i
            if line['type'] == 'q':
                refs = line['refs']
                f1 = refs[0]-1
                f2 = refs[1]-1 if len(refs) > 1 else -1
                id = line['id']
                indices = [idx for idx in range(i-id+1, i+1)]# if lines[idx]['type'] != 'q' or idx == i]
                memory_list = L[indices]

                m_o1 = O_t([id-1], memory_list, s_Ot)
                if f2 != -1:
                    m_o2 = O_t([id-1, m_o1], memory_list, s_Ot)
                else:
                    m_o2 = -1

                err = train(id-1, f1, f2, H[line['answer']], m_o1, m_o2, gamma, memory_list, V)[0]
                total_err += err
        print "epoch: ", epoch, " err: ", (total_err/len(lines))
    return U_Ot, U_R, V, H, phi_x1, phi_x2, phi_y, phi_t, s_Ot, sR

def do_test(fname, vectorizer, U_Ot, U_R, V, H, phi_x1, phi_x2, phi_y, phi_t, s_Ot, sR):
    lenW = len(vectorizer.vocabulary_)
    lines = get_lines(fname)
    L = vectorizer.transform([x['text'] for x in lines]).toarray().astype(np.float32)

    W = 3*lenW
    Y_true = []
    Y_pred = []
    for i,line in enumerate(lines):
        if line['type'] == 'q':
            r = line['answer']
            id = line['id']
            indices = [idx for idx in range(i-id+1, i+1)]# if lines[idx]['type'] != 'q' or idx == i]
            memory_list = L[indices]

            m_o1 = O_t([id-1], memory_list, s_Ot)
            m_o2 = O_t([id-1, m_o1], memory_list, s_Ot)

            bestVal = None
            best = None
            for w in vectorizer.vocabulary_:
                val = sR([id-1, m_o1, m_o2], H[w], memory_list, V)
                if bestVal is None or val > bestVal:
                    bestVal = val
                    best = w
            Y_true.append(r)
            Y_pred.append(best)
    print metrics.classification_report(Y_true, Y_pred)

def main():
    vectorizer = CountVectorizer()
    U_Ot, U_R, V, H, phi_x1, phi_x2, phi_y, phi_t, s_Ot, sR = do_train(TRAIN_FILE, vectorizer)
    do_test(TEST_FILE, vectorizer, U_Ot, U_R, V, H, phi_x1, phi_x2, phi_y, phi_t, s_Ot, sR)

main()
