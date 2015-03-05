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

TRAIN_FILE='data/en/qa2_two-supporting-facts_train.txt'
TEST_FILE='data/en/qa2_two-supporting-facts_test.txt'
TRAIN_FILE = sys.argv[1]
TEST_FILE = sys.argv[2]
#print "train: ", TRAIN_FILE
#print "test: ", TEST_FILE

D = 50
gamma = 0.1

def zeros(shape, dtype=np.float32):
    return np.zeros(shape, dtype)

def O_t(xs, L, s):
    t = len(L)-2
    for i in xrange(len(L)-3, -1, -1):
        if s(xs, i, t, L) > 0:
            t = i
    return t

def get_train(U_Ot, U_R, lenW, n_facts):
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
            lambda x_t, t: T.switch(T.eq(t, T.shape(L)[0]-1), 0,
                T.dot(T.dot(T.switch(T.eq(t, 0), phi_x1(x_t, L).reshape((1,-1)), phi_x2(x_t, L).reshape((1,-1))), U_Ot.T),
                T.dot(U_Ot, (phi_y(y_t, L) - phi_y(yp_t, L) + phi_t(x_t, y_t, yp_t, L))))),
            sequences=[xs, T.arange(T.shape(xs)[0])])
        return result.sum()
    def sR(xs, y_t, L, V):
        result, updates = theano.scan(
            lambda x_t, t: T.dot(T.dot(T.switch(T.eq(t, 0), phi_x1(x_t, L).reshape((1,-1)), phi_x2(x_t, L).reshape((1,-1))), U_R.T),
                                 T.dot(U_R, phi_y(y_t, V))),
            sequences=[xs, T.arange(T.shape(xs)[0])])
        return result.sum()

    x_t = T.iscalar('x_t')
    m = [x_t] + [T.iscalar('m_o%d' % i) for i in xrange(n_facts)]
    f = [T.iscalar('f%d_t' % i) for i in xrange(n_facts)]
    r_t = T.iscalar('r_t')
    gamma = T.scalar('gamma')
    L = T.fmatrix('L') # list of messages
    V = T.fmatrix('V') # vocab
    r_args = T.stack(*([x_t] + m))

    cost_arr = [0] * 2 * (len(m)-1)
    updates_arr = [0] * 2 * (len(m)-1)
    for i in xrange(len(m)-1):
        cost_arr[2*i], updates_arr[2*i] = theano.scan(
                lambda f_bar, t: T.largest(gamma - s_Ot(T.stack(*m[:i+1]), f[i], t, L), 0),
            sequences=[L, T.arange(T.shape(L)[0])])
        cost_arr[2*i+1], updates_arr[2*i+1] = theano.scan(
                lambda f_bar, t: T.largest(gamma + s_Ot(T.stack(*m[:i+1]), f[i], t, L), 0),
            sequences=[L, T.arange(T.shape(L)[0])])

    cost1, u1 = theano.scan(
        lambda r_bar, t: T.largest(gamma - sR(r_args, r_t, L, V) + sR(r_args, t, L, V), 0),
        sequences=[V, T.arange(T.shape(V)[0])])

    cost = cost1.sum() - (2*(len(m)-1) + 1) * gamma
    for c in cost_arr:
        cost += c.sum()

    g_uo, g_ur = T.grad(cost, [U_Ot, U_R])

    train = theano.function(
        inputs=[r_t, gamma, L, V] + m + f,
        outputs=[cost],
        updates=[(U_Ot, U_Ot-0.001*g_uo), (U_R, U_R-0.001*g_ur)])
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

def do_train(lines, L, vectorizer):
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
            if i == len(xs)-1:
                continue
            x = phi_x1(x_t, L) if i == 0 else phi_x2(x_t, L)
            y = phi_y(y_t, L)
            yp = phi_y(yp_t, L)
            U = U_Ot.get_value()
            result += np.dot(np.dot(x.reshape((1,-1)), U.T), np.dot(U, y - yp + phi_t(x_t, y_t, yp_t, L)))
        return result

    lenW = len(vectorizer.vocabulary_)
    H = {}
    for i,v in enumerate(vectorizer.vocabulary_):
        H[v] = i
    V = vectorizer.transform([v for v in vectorizer.vocabulary_]).toarray().astype(np.float32)

    W = 3*lenW + 3
    U_Ot = theano.shared(np.random.randn(D, W).astype(np.float32))
    U_R = theano.shared(np.random.randn(D, W).astype(np.float32))
    train = None

    for epoch in range(100):
        total_err = 0
        print "*" * 80
        print "epoch: ", epoch
        n_wrong = 0
        for i,line in enumerate(lines):
            if i % 1000 == 0:
                print "i: ", i, " nwrong: ", n_wrong
            if line['type'] == 'q':
                refs = line['refs']
                f = [ref - 1 for ref in refs]
                id = line['id']-1
                offset = i-id
                indices = [idx for idx in range(offset-1, i+1) if lines[idx]['type'] != 'q' or idx == i]
                memory_list = L[indices]
                orig = lines[indices]
                id = indices.index(offset+id)
                f = [indices.index(offset+ref) for ref in f]
                m = []
                for j in xrange(len(refs)):
                    m.append(O_t([id]+f[:j], memory_list, s_Ot))

                if m[0] != f[0]:
                    n_wrong += 1

                if train is None:
                    train = get_train(U_Ot, U_R, lenW, len(refs))
                err = train(H[line['answer']], gamma, memory_list, V, id, *(m + f))[0]
                total_err += err
        print "epoch: ", epoch, " err: ", (total_err/len(lines))
    return U_Ot, U_R, V, H, phi_x1, phi_x2, phi_y, phi_t, s_Ot, sR

def do_test(lines, L, vectorizer, U_Ot, U_R, V, H, phi_x1, phi_x2, phi_y, phi_t, s_Ot, sR):
    lenW = len(vectorizer.vocabulary_)
    W = 3*lenW
    Y_true = []
    Y_pred = []
    for i,line in enumerate(lines):
        if line['type'] == 'q':
            r = line['answer']
            id = line['id']-1
            offset = i-id
            indices = [idx for idx in range(offset-1, i+1) if lines[idx]['type'] != 'q' or idx == i]
            memory_list = L[indices]
            id = indices.index(offset+id)

            m_o1 = O_t([id], memory_list, s_Ot)
            m_o2 = O_t([id, m_o1], memory_list, s_Ot)

            bestVal = None
            best = None
            for w in vectorizer.vocabulary_:
                val = sR([id, m_o1, m_o2], H[w], memory_list, V)
                if bestVal is None or val > bestVal:
                    bestVal = val
                    best = w
            Y_true.append(r)
            Y_pred.append(best)
    print metrics.classification_report(Y_true, Y_pred)

def main():
    train_lines, test_lines = get_lines(TRAIN_FILE), get_lines(TEST_FILE)
    lines = np.concatenate([train_lines, test_lines], axis=0)
    vectorizer = CountVectorizer()
    vectorizer.fit([x['text'] + ' ' + x['answer'] if 'answer' in x else x['text'] for x in lines])
    L = vectorizer.transform([x['text'] for x in lines]).toarray().astype(np.float32)
    L_train, L_test = L[xrange(len(train_lines))], L[xrange(len(test_lines),len(lines))]
    U_Ot, U_R, V, H, phi_x1, phi_x2, phi_y, phi_t, s_Ot, sR = do_train(train_lines, L_train, vectorizer)
    do_test(test_lines, L_test, vectorizer, U_Ot, U_R, V, H, phi_x1, phi_x2, phi_y, phi_t, s_Ot, sR)

main()
