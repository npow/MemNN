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

def getBest(L, memory_list, s):
    best = None
    bestVal = None
    for f_bar in memory_list:
        val = s(L, f_bar)
        if bestVal is None or val > bestVal:
            best = f_bar
            bestVal = val
    return best

def get_train(U_O, U_R, lenW):
    def s(x, y, U):
        return T.dot(T.dot((x).reshape((1,-1)), U.T), T.dot(U, (y)))
    def phi_x1(x):
        return T.concatenate([x, np.zeros((2*lenW,)).astype(np.float32)], axis=0)
    def phi_x2(x):
        return T.concatenate([np.zeros((lenW,)).astype(np.float32), x, np.zeros((lenW,)).astype(np.float32)], axis=0)
    def phi_y(x):
        return T.concatenate([np.zeros((2*lenW,)).astype(np.float32), x], axis=0)

    m_o1 = T.fvector('m_o1')
    m_o2 = T.fvector('m_o2')
    x = T.fvector('x')
    y = T.fvector('y')
    r = T.fvector('r')
    gamma = T.scalar('gamma')
    f1 = T.fvector('f1')
    f2 = T.fvector('f2')
    L = T.fmatrix('L') # list of messages
    V = T.fmatrix('V') # vocab
    sO = partial(s, U=U_O)
    sR = partial(s, U=U_R)
    cost1, u1 = theano.scan(lambda f_bar: T.largest(gamma - sO(phi_x1(x), phi_y(f1)) + sO(phi_x1(x), phi_y(f_bar)), 0), L)
    cost2, u2 = theano.scan(lambda f_bar: T.largest(gamma - sO(phi_x1(x), phi_y(f2)) - sO(phi_x2(m_o1), phi_y(f2)) + sO(phi_x1(x), phi_y(f_bar)) + sO(phi_x2(m_o1), phi_y(f_bar)), 0), L)
    cost3, u3 = theano.scan(lambda r_bar: T.largest(gamma - sR(phi_x1(x), phi_y(r)) - sR(phi_x2(m_o1), phi_y(r)) - sR(phi_x2(m_o2), phi_y(r)) + sR(phi_x1(x), phi_y(r_bar)) + sR(phi_x2(m_o1), phi_y(r_bar)) + sR(phi_x2(m_o2), phi_y(r_bar)), 0), V)
    cost = cost1.sum() + cost2.sum() + cost3.sum() - 3*gamma
    g_uo, g_ur = T.grad(cost, [U_O, U_R])

    train = theano.function(
        inputs=[m_o1, m_o2, x, gamma, f1, f2, L, V, r],
        outputs=[cost],
        updates=[(U_O, U_O-0.01*g_uo), (U_R, U_R-0.01*g_ur)])
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
    def phi_x1(x):
        return np.concatenate([x, np.zeros((2*lenW,)).astype(np.float32)], axis=0)
    def phi_x2(x):
        return np.concatenate([np.zeros((lenW,)).astype(np.float32), x, np.zeros((lenW,)).astype(np.float32)], axis=0)
    def phi_y(y):
        return np.concatenate([np.zeros((2*lenW,)).astype(np.float32), y], axis=0)
    def s(x, y, U):
        return np.dot(np.dot(x.reshape((1,-1)), U.T), np.dot(U, y))
    def sO(L, y):
        result = 0
        for i,x in enumerate(L):
            if i == 0:
                result += s(phi_x1(x), phi_y(y), U_O.get_value())
            else:
                result += s(phi_x2(x), phi_y(y), U_O.get_value())
        return result
    def sR(L, y):
        result = 0
        for i,x in enumerate(L):
            if i == 0:
                result += s(phi_x1(x), phi_y(y), U_R.get_value())
            else:
                result += s(phi_x2(x), phi_y(y), U_R.get_value())
        return result

    lines = get_lines(fname)
    L = vectorizer.fit_transform([x['text'] for x in lines]).toarray().astype(np.float32)
    lenW = len(vectorizer.vocabulary_)
    H = {}
    for i,v in enumerate(vectorizer.vocabulary_):
        H[v] = i
    V = vectorizer.transform([v for v in vectorizer.vocabulary_]).toarray().astype(np.float32)

    W = 3*lenW
    U_O = theano.shared(np.random.randn(D, W).astype(np.float32))
    U_R = theano.shared(np.random.randn(D, W).astype(np.float32))
    train = get_train(U_O, U_R, lenW)

    for epoch in range(10):
        err = 0
        print "*" * 80
        print "epoch: ", epoch
        bar = pyprind.ProgBar(len(lines), monitor=True)
        for i,line in enumerate(lines):
            if line['type'] == 'q':
                refs = line['refs']
                f1 = refs[0]
                f2 = refs[1] if len(refs) > 1 else None
                if f1 is not None:
                    f1 = L[f1-1]
                if f2 is not None:
                    f2 = L[f2-1]
                else:
                    f2 = np.zeros((lenW,)).astype(np.float32)
                r = V[H[line['answer']]].T
                id = line['id']
                indices = [idx for idx in range(i-id+1, i+1) if lines[idx]['type'] != 'q']
                memory_list = L[indices]

                x = L[i]
                m_o1 = getBest([x], memory_list, sO)
                if f2 is not None:
                    m_o2 = getBest([x, m_o1], memory_list, sO)
                else:
                    m_o2 = np.zeros((lenW,)).astype(np.float32)

                err += train(m_o1, m_o2, x, gamma, f1, f2, memory_list, V, r)[0]
            bar.update()
        print "epoch: ", epoch, " err: ", (err/len(lines))
    return U_O, U_R, V, H, phi_x1, phi_x2, phi_y, sO, sR, s

def do_test(fname, vectorizer, U_O, U_R, V, H, phi_x1, phi_x2, phi_y, sO, sR, s):
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
            indices = [idx for idx in range(i-id+1, i+1) if lines[idx]['type'] != 'q']
            memory_list = L[indices]

            x = L[i]
            m_o1 = getBest([x], memory_list, sO)
            m_o2 = getBest([x, m_o1], memory_list, sO)

            bestVal = None
            best = None
            for w in vectorizer.vocabulary_:
                val = sR([x, m_o1, m_o2], V[H[w]].T)
                if bestVal is None or val > bestVal:
                    bestVal = val
                    best = w
            Y_true.append(r)
            Y_pred.append(best)
    print metrics.classification_report(Y_true, Y_pred)

def main():
    vectorizer = CountVectorizer()
    U_O, U_R, V, H, phi_x1, phi_x2, phi_y, sO, sR, s = do_train(TRAIN_FILE, vectorizer)
    do_test(TEST_FILE, vectorizer, U_O, U_R, V, H, phi_x1, phi_x2, phi_y, sO, sR, s)

main()
