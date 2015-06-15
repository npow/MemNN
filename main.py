from __future__ import division
import argparse
import numpy as np
import sys
from collections import OrderedDict
from sklearn import metrics
from sklearn.feature_extraction.text import *
from sklearn.preprocessing import *
from theano.ifelse import ifelse
import theano
import theano.tensor as T

TRAIN_FILE='data/en-10k/qa1_single-supporting-fact_train.txt'
TEST_FILE='data/en-10k/qa1_single-supporting-fact_test.txt'

def zeros(shape, dtype=np.float32):
    return np.zeros(shape, dtype)

# TODO: convert this to a theano function
def O_t(xs, L, s):
    t = 0
    for i in xrange(len(L)-1): # last element is the answer, so we can skip it
        if s(xs, i, t, L) > 0:
            t = i
    return t

def sgd(cost, params, learning_rate):
    grads = T.grad(cost, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad

    return updates

class Model:
    def __init__(self, train_file, test_file, D=50, gamma=1, lr=0.001):
        self.train_lines, self.test_lines = self.get_lines(train_file), self.get_lines(test_file)
        lines = np.concatenate([self.train_lines, self.test_lines], axis=0)

        self.vectorizer = CountVectorizer(lowercase=False)
        self.vectorizer.fit([x['text'] + ' ' + x['answer'] if 'answer' in x else x['text'] for x in lines])

        L = self.vectorizer.transform([x['text'] for x in lines]).toarray().astype(np.float32)
        self.L_train, self.L_test = L[:len(self.train_lines)], L[len(self.train_lines):]

        self.train_model = None
        self.D = D
        self.gamma = gamma
        self.lr = lr
        self.H = None
        self.V = None

    def create_train(self, lenW, n_facts):
        ONE = theano.shared(np.float32(1))
        ZERO = theano.shared(np.float32(0))
        def phi_x1(x_t, L):
            return T.concatenate([L[x_t].reshape((-1,)), zeros((2*lenW,)), zeros((3,))], axis=0)
        def phi_x2(x_t, L):
            return T.concatenate([zeros((lenW,)), L[x_t].reshape((-1,)), zeros((lenW,)), zeros((3,))], axis=0)
        def phi_y(x_t, L):
            return T.concatenate([zeros((2*lenW,)), L[x_t].reshape((-1,)), zeros((3,))], axis=0)
        def phi_t(x_t, y_t, yp_t, L):
            return T.concatenate([zeros(3*lenW,), T.stack(T.switch(T.lt(x_t,y_t), ONE, ZERO), T.switch(T.lt(x_t,yp_t), ONE, ZERO), T.switch(T.lt(y_t,yp_t), ONE, ZERO))], axis=0)
        def s_Ot(xs, y_t, yp_t, L):
            result, updates = theano.scan(
                lambda x_t, t: T.dot(T.dot(T.switch(T.eq(t, 0), phi_x1(x_t, L).reshape((1,-1)), phi_x2(x_t, L).reshape((1,-1))), self.U_Ot.T),
                               T.dot(self.U_Ot, (phi_y(y_t, L) - phi_y(yp_t, L) + phi_t(x_t, y_t, yp_t, L)))),
                sequences=[xs, T.arange(T.shape(xs)[0])])
            return result.sum()
        def sR(xs, y_t, L, V):
            result, updates = theano.scan(
                lambda x_t, t: T.dot(T.dot(T.switch(T.eq(t, 0), phi_x1(x_t, L).reshape((1,-1)), phi_x2(x_t, L).reshape((1,-1))), self.U_R.T),
                                     T.dot(self.U_R, phi_y(y_t, V))),
                sequences=[xs, T.arange(T.shape(xs)[0])])
            return result.sum()
            
        x_t = T.iscalar('x_t')
        y_t = T.iscalar('y_t')
        yp_t = T.iscalar('yp_t')
        xs = T.ivector('xs')
        m = [x_t] + [T.iscalar('m_o%d' % i) for i in xrange(n_facts)]
        f = [T.iscalar('f%d_t' % i) for i in xrange(n_facts)]
        r_t = T.iscalar('r_t')
        gamma = T.scalar('gamma')
        L = T.fmatrix('L') # list of messages
        V = T.fmatrix('V') # vocab
        r_args = T.stack(*m)

        cost_arr = [0] * 2 * (len(m)-1)
        for i in xrange(len(m)-1):
            cost_arr[2*i], _ = theano.scan(
                    lambda f_bar, t: T.switch(T.or_(T.eq(t, f[i]), T.eq(t, T.shape(L)[0]-1)), 0, T.largest(gamma - s_Ot(T.stack(*m[:i+1]), f[i], t, L), 0)),
                sequences=[L, T.arange(T.shape(L)[0])])
            cost_arr[2*i+1], _ = theano.scan(
                    lambda f_bar, t: T.switch(T.or_(T.eq(t, f[i]), T.eq(t, T.shape(L)[0]-1)), 0, T.largest(gamma + s_Ot(T.stack(*m[:i+1]), t, f[i], L), 0)),
                sequences=[L, T.arange(T.shape(L)[0])])

        cost1, _ = theano.scan(
            lambda r_bar, t: T.switch(T.eq(r_t, t), 0, T.largest(gamma - sR(r_args, r_t, L, V) + sR(r_args, t, L, V), 0)),
            sequences=[V, T.arange(T.shape(V)[0])])

        cost = cost1.sum()
        for c in cost_arr:
            cost += c.sum()

        updates = sgd(cost, [self.U_Ot, self.U_R], learning_rate=self.lr)

        self.train_model = theano.function(
            inputs=[r_t, gamma, L, V] + m + f,
            outputs=[cost],
            updates=updates)

        self.sR = theano.function([xs, y_t, L, V], sR(xs, y_t, L, V))
        self.s_Ot = theano.function([xs, y_t, yp_t, L], s_Ot(xs, y_t, yp_t, L))
        
    def train(self, n_epochs):
        lenW = len(self.vectorizer.vocabulary_)
        self.H = {}
        for i,v in enumerate(self.vectorizer.vocabulary_):
            self.H[v] = i
        self.V = self.vectorizer.transform([v for v in self.vectorizer.vocabulary_]).toarray().astype(np.float32)

        W = 3*lenW + 3
        self.U_Ot = theano.shared(np.random.uniform(-0.1, 0.1, (self.D, W)).astype(np.float32))
        self.U_R = theano.shared(np.random.uniform(-0.1, 0.1, (self.D, W)).astype(np.float32))

        for epoch in range(n_epochs):
            total_err = 0
            print "*" * 80
            print "epoch: ", epoch
            n_wrong = 0
            
            for i,line in enumerate(self.train_lines):
                if i > 0 and i % 1000 == 0:
                    print "i: ", i, " nwrong: ", n_wrong
                if line['type'] == 'q':
                    refs = line['refs']
                    f = [ref - 1 for ref in refs]
                    id = line['id']-1
                    indices = [idx for idx in range(i-id, i+1)]
                    memory_list = self.L_train[indices]
#                    print "REFS: ", self.train_lines[indices][f], "\nMEMORY: ", self.train_lines[indices], '\n', '*' * 80

                    if self.train_model is None:
                        self.create_train(lenW, len(f))                    

                    m = f
                    mm = []
                    for j in xrange(len(f)):
                        mm.append(O_t([id]+m[:j], memory_list, self.s_Ot))

                    if mm[0] != f[0]:
                        n_wrong += 1
                        
                    err = self.train_model(self.H[line['answer']], self.gamma, memory_list, self.V, id, *(m + f))[0]
                    total_err += err
            print "i: ", i, " nwrong: ", n_wrong
            print "epoch: ", epoch, " err: ", (total_err/len(self.train_lines))

            # TODO: use validation set
            self.test()
        
    def test(self):
        lenW = len(self.vectorizer.vocabulary_)
        W = 3*lenW
        Y_true = []
        Y_pred = []
        for i,line in enumerate(self.test_lines):
            if line['type'] == 'q':
                r = line['answer']
                id = line['id']-1
                indices = [idx for idx in range(i-id, i+1)]
                memory_list = self.L_test[indices]

                m_o1 = O_t([id], memory_list, self.s_Ot)
                m_o2 = O_t([id, m_o1], memory_list, self.s_Ot)

                bestVal = None
                best = None
                for w in self.vectorizer.vocabulary_:
                    val = self.sR([id, m_o1, m_o2], self.H[w], memory_list, self.V)
                    if bestVal is None or val > bestVal:
                        bestVal = val
                        best = w
                Y_true.append(r)
                Y_pred.append(best)
        print metrics.classification_report(Y_true, Y_pred)

    def get_lines(self, fname):
        lines = []
        for i,line in enumerate(open(fname)):
            id = int(line[0:line.find(' ')])
            line = line.strip()
            line = line[line.find(' ')+1:]        
            if line.find('?') == -1:
                lines.append({'type':'s', 'text': line})
            else:
                idx = line.find('?')
                tmp = line[idx+1:].split('\t')
                lines.append({'id':id, 'type':'q', 'text': line[:idx], 'answer': tmp[1].strip(), 'refs': [int(x) for x in tmp[2:][0].split(' ')]})
            if False and i > 1000:
                break
        return np.array(lines)

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def main():
    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)
    parser.add_argument('--train_file', type=str, default=TRAIN_FILE, help='Train file')
    parser.add_argument('--test_file', type=str, default=TEST_FILE, help='Test file')
    parser.add_argument('--gamma', type=float, default=1, help='Gamma')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--embedding_size', type=int, default=50, help='Embedding size')
    parser.add_argument('--n_epochs', type=int, default=10, help='Num epochs')
    args = parser.parse_args()
    print "args: ", args

    model = Model(args.train_file, args.test_file, D=args.embedding_size, gamma=args.gamma, lr=args.lr)
    model.train(args.n_epochs)

if __name__ == '__main__':
    main()
