import os
import inspect
import pickle
try:
    import cPickle
except:
    pass
import numpy as np


def disp(val):
    if type(val) is type:
        return str(val[1])+'\n'
    if (type(val) is list or type(val) is np.ndarray) and len(val) == 1:
        return str(val[0])+'\n'
    return str(val)+'\n'


def load_pickle(path):
    try:
        import msgpack
        import msgpack_numpy as m
        m.patch()
        return msgpack.unpack(open(path, 'r'))
    except:
        try:
            return pickle.load(open(path, 'rb'), fix_imports=True, encoding="latin1")
        except TypeError:
            try:
                return pickle.load(open(path, 'r'))
            except ValueError:
                return cPickle.loads(path)


def save_pickle(path, obj):
    import msgpack
    import msgpack_numpy as m
    m.patch()
    return msgpack.pack(obj, open(path, 'w'))


def load_csv(pkl, path):
    speakers = {s: None for s in pkl['speaker']}
    with open(path, 'r') as fin:
        lines = fin.readlines()
        header = lines[0].split()
        lines = [line.split() for line in lines[1:]]
        spk_dct = {l[0]: None for l in lines}
        for j in range(1, len(header)):
            feat = header[j]
            if feat not in pkl:
                dct = {line[0]: float(line[j]) for line in lines}
                for s in speakers:
                    if s not in spk_dct:
                        dct[s] = None
                pkl[feat] = np.array([dct[s] for s in pkl['speaker']])


def print_results(y, y_act, y_exp, args, speakers=None, feats=None, eval_pdfs=None, a_out=None, a_list=None):
    if eval_pdfs is not None:
        with open(args.OUT_PATH + '/deep_' + feats + '_features.txt', 'w') as fout:
            fout.write('\n'.join([speakers[k] + ' ' + ' '.join([str(j) for j in eval_pdfs[k]]) for k in range(
                len(eval_pdfs))]))
    if a_out is not None:
        for a_ind in range(len(a_list)):
            a_name = a_list[a_ind]
            a_val = a_out[a_ind]
            save_pickle(args.OUT_PATH + '/' + a_name + '.pkl', a_val)
    if y is not None:
        y = np.reshape(y, [-1])
        y_act = np.reshape(y_act, [-1])
        eval_mse = str(np.mean(np.power(y - y_act, 2)))
        eval_acc = np.corrcoef(y, y_act)[0][1]
        print('Accuracy is', disp(eval_acc), 'mse ' + eval_mse, ' on BULATS graders')

        if not os.path.exists(args.OUT_PATH + '/against-original'):
            os.mkdir(args.OUT_PATH + '/against-original')

        with open(args.OUT_PATH + '/against-original/pearson.txt', 'w') as fout:
            fout.write(disp(eval_acc))
        with open(args.OUT_PATH + '/against-original/mse.txt', 'w') as fout:
            fout.write(str(eval_mse) + '\n')
        with open(args.OUT_PATH + '/against-original/references.txt', 'w') as fout:
            fout.write('\n'.join([
                ('' if speakers is None else (speakers[i] + ' ')) + str(y_act[i]) for i in range(len(y_act))]))
        with open(args.OUT_PATH + '/against-original/predictions.txt', 'w') as fout:
            fout.write('\n'.join([('' if speakers is None else (speakers[i] + ' ')) + str(y[i]) for i in range(len(y))]))
        with open(args.OUT_PATH + '/against-original/ref_pred.txt', 'w') as fout:
            fout.write('\n'.join([
                ('' if speakers is None else (speakers[i] + ' ')
                 ) + str(y_act[i]) + ' ' + str(y[i]) for i in range(len(y))]))

    if y_exp is not None:
        exp_mse = str(np.mean(np.power(y - y_exp, 2)))
        exp_acc = np.corrcoef(y, y_exp)[0][1]
        print('Accuracy is', disp(exp_acc), 'mse ' + exp_mse, ' on expert graders')
        os.mkdir(args.OUT_PATH + '/against-expert')
        with open(args.OUT_PATH + '/against-expert/pearson.txt', 'w') as fout:
            fout.write(disp(exp_acc))
        with open(args.OUT_PATH + '/against-expert/mse.txt', 'w') as fout:
            fout.write(str(exp_mse) + '\n')
        with open(args.OUT_PATH + '/against-expert/references.txt', 'w') as fout:
            fout.write('\n'.join([('' if speakers is None else (speakers[i] + ' ')
                                   ) + str(y_exp[i]) for i in range(len(y_exp))]))
        with open(args.OUT_PATH + '/against-expert/predictions.txt', 'w') as fout:
            fout.write('\n'.join([('' if speakers is None else (speakers[i] + ' ')
                                   ) + str(y[i]) for i in range(len(y))]))
        with open(args.OUT_PATH + '/against-original/ref_pred.txt', 'w') as fout:
            fout.write('\n'.join([('' if speakers is None else (speakers[i] + ' ')
                                   ) + ' ' + str(y_exp[i]) + ' ' + str(y[i]) for i in range(len(y))]))


def read_pickles(in_paths):
    big_pkl = {}
    paths = [p for p in in_paths if p.endswith('.pkl')]
    for p in range(len(paths)):
        if not os.path.isdir(paths[p]) and paths[p].endswith('.pkl'):
            path = paths[p]
            pkl = load_pickle(path)
            lens = [len(pkl[k]) for k in pkl]
            if len(list(set(lens))) != 1:
                raise ValueError('Inconsistent numbers of speakers ' + str(lens))
            if len(big_pkl) == 0:
                for k in pkl:
                    big_pkl[k] = pkl[k]
            else:
                for k in big_pkl:
                    if k not in pkl:
                        raise ValueError('Key ' + str(k) + ' not present in pickle ' + path)
                for k in pkl:
                    if k not in big_pkl:
                        raise ValueError('Key ' + k + ' in ' + path + ' not present in ' + str(path[:p]))
                    if type(pkl[k]) != type(big_pkl[k]):
                        raise TypeError(
                            'Entry ' + k + ' in ' + path + ': expected ' + str(type(big_pkl[k])) + ' - but got ' + str(
                                type(pkl[k])))
                    if type(pkl[k]) is dict:
                        for j in pkl[k]:
                            big_pkl[k][len(big_pkl[k])] = pkl[k][j]
                    elif type(pkl[k]) is list:
                        big_pkl[k] += pkl[k]
                    elif type(pkl[k]) is np.ndarray:
                        big_pkl[k] = np.concatenate((big_pkl[k], pkl[k]))
                    else:
                        raise ValueError('Invalid type ' + str(type(pkl[k])) + ' for key ' + k)
    for path in in_paths:
        if path.endswith('.txt'):
            load_csv(big_pkl, path)
    return big_pkl


def norm(X_raw, feat, mean=None, std=None):
    vals = [val for val in X_raw[:, feat] if val > 0]
    if len(vals) == 0:
        mean = 0.0
        std = 1.0
    else:
        try:
            mean = np.mean(vals) if mean is None else (mean if type(mean) is float or type(mean) is int else mean[feat])
        except Exception as e:
            print(feat, len(vals), len(vals[0]), len(vals[0][0]))
            raise e
        std = np.std(vals) if std is None else (std if type(std) is float or type(std) is int else std[feat])
    Xf_norm = [(-1.0 if val is None else val) if val != 1.0 else (val - mean) / std for val in X_raw[:, feat]]
    return Xf_norm, mean, std


def get_arg_spec(func_or_init):
    """
    Returns the argument specs of a function or of the initialiser of a class
    :param func_or_init: Function or class
    :return:
    """
    try:
        return inspect.getargspec(func_or_init)[0]
    except TypeError:
        return inspect.getargspec(func_or_init.__init__)[0]


def opt_arg(func, args):
    """
    Calls function with dictionary of arguments applied optionally
    :param func: Function
    :param args: Arguments
    :return:
    """
    return func(**{key: args[key] for key in args if key in get_arg_spec(func)})


def batch(x, y, batch_size=None, lengths=None, max_len=None, l1s=None, index=None, xtype=np.float64, ytype=np.float64):
    if batch_size is None or batch_size > len(x):
        batch_size = len(x)
    if lengths is not None:
        X = [np.zeros((batch_size, max_len, x[0].shape[1]), dtype=xtype) for j in range(len(x) // batch_size)]
        l = [np.zeros(batch_size, np.int64) for j in range(len(x) // batch_size)]
    else:
        X = [np.zeros((batch_size, x[0].shape[0]), dtype=xtype) for j in range(len(x) // batch_size)]
    Y = [np.zeros(batch_size,dtype=ytype) for j in range(len(x) // batch_size)]
    if l1s is not None:
        l1 = [np.zeros(batch_size, np.int64) for j in range(len(x) // batch_size)]
    for i in range(len(x)):
        if i // batch_size < len(x) // batch_size:
            Y[i // batch_size][i % batch_size] = y[i]
            if lengths is not None:
                X[i // batch_size][i % batch_size][:lengths[i]] = x[i]
                l[i // batch_size][i % batch_size] = lengths[i]
            else:
                X[i // batch_size][i % batch_size] = x[i]
            if l1s is not None:
                l1[i // batch_size][i % batch_size] = l1s[i]
    if lengths is not None:
        if l1s is None:
            if index is None:
                return X, l, Y
            else:
                return X[index], l[index], Y[index]
        return X, l, Y, l1
    if l1s is None:
        return X, Y
    return X, Y, l1


def readlines(path):
    with open(path, 'r') as fin:
        return fin.readlines()


def get_from_txt(path, in_vals, in_col=0, get_col=1, sep=' ', skip_header=False):
    dct = {line.split()[in_col]: line.split()[get_col] for line in readlines(path)[1 if skip_header else 0:]}
    return [float(dct[val]) for val in in_vals]


def get_length(X):
    return np.array([X[j].shape[0] for j in range(len(X))], np.int64)


def dict2np(X):
    return np.array([X[k] for k in X], np.int64).reshape([-1, 1])
