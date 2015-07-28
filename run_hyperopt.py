import argparse
import glob
from hyperopt import hp, fmin, tpe, STATUS_OK
from main import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, default=1, help='Task#')
    parser.add_argument('--train_file', type=str, default='', help='Train file')
    parser.add_argument('--test_file', type=str, default='', help='Test file')
    parser.add_argument('--n_epochs', type=int, default=10, help='Num epochs')
    parser.add_argument('--max_evals', type=int, default=100, help='Max evals')
    args = parser.parse_args()
    print "args: ", args

    train_file = glob.glob('data/en-10k/qa%d_*.txt' % args.task)[0]
    test_file = glob.glob('data/en-10k/qa%d_*.txt' % args.task)[0]
    if args.train_file != '' and args.test_file != '':
        train_file, test_file = args.train_file, args.test_file

    print 'train_file:', train_file
    print 'test_file:', test_file

    space = {
        'embedding_size': hp.quniform('embedding_size', 10, 500, 50),
        'lr': hp.uniform('lr', .001, .1),
        'gamma': hp.lognormal('gamma', .001, 10)
    }

    def objective(params):
        print params
        model = Model(train_file, test_file, D=int(params['embedding_size']), gamma=params['gamma'], lr=params['lr'])
        err = model.train(args.n_epochs)
        return { 'loss': err, 'status': STATUS_OK }

    best = fmin(objective, space=space, algo=tpe.suggest, max_evals=args.max_evals)

    model = Model(train_file, test_file, D=best['embedding_size'], gamma=best['gamma'], lr=best['lr'])
    model.train(args.n_epochs)
