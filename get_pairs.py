import os
import math
import pickle
import random
import argparse
from util import read_pickles
from sys import argv
import operator as op
from functools import reduce


def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer * 1.0 / denom


def get_phones(alphabet):
    if alphabet == 'arpabet':
        vowels = ['aa', 'ae', 'eh', 'ah', 'ea', 'ao', 'ia', 'ey', 'aw', 'ay', 'ax', 'er', 'ih', 'iy',
          'uh', 'oh', 'oy', 'ow', 'ua', 'uw']
        consonants = ['el', 'ch', 'en', 'ng', 'sh', 'th', 'zh', 'w', 'dh', 'hh', 'jh', 'em', 'b', 'd',
              'g', 'f', 'h', 'k', 'm', 'l', 'n', 'p', 's', 'r', 't', 'v', 'y', 'z']+['sil']
        phones = vowels+consonants
        return vowels, consonants, phones
    if alphabet == 'graphemic':
        vowels = ['a','e','i','o','u']
        consonants = ['b','c','d','f','g','h','j','k','l','m','n','o','p','q','r','s','t','v','w','x','y','z']+['sil']
        phones = vowels+consonants
        return vowels, consonants, phones
    with open(alphabet,'r') as fin:
        return None,None, [l.split()[0] for l in fin.readlines()]


def get_phone_instances(pkl):
    instances = {}
    for spk in range(len(pkl['plp'])):
        instances[spk] = {}
        for utt in range(len(pkl['plp'][spk])):
            for w in range(len(pkl['plp'][spk][utt])):
                for ph in range(len(pkl['plp'][spk][utt][w])):
                    phone_label = pkl['phone'][spk][utt][w][ph]
                    if phone_label not in instances[spk]:
                        instances[spk][phone_label] = {}
                    instances[spk][phone_label][len(instances[spk][phone_label])] = pkl['plp'][spk][utt][w][ph]
    return instances


def get_pairs(instance_dict,N_total):
    pairs = {}
    for spk in instance_dict:
        instances = instance_dict[spk]
        if len(instances) > 0:
            N = N_total*1.0/len(instance_dict)
            ks = [k for k in instances]
            f = sum([ncr(len(instances[ph]),2) for ph in instances])*1.0/N
            for ph in instances:
                n = int(math.ceil(ncr(len(instances[ph]), 2)*1.0/f))
                ln = len(instances[ph])
                if ln < n:
                    for j in range(ln):
                        for i in random.sample(range(j+1,ln), min(int(round(n*1.0/ln)),ln-j-1)):
                            pairs[len(pairs)] = (instances[ph][j],instances[ph][i],0)
                else:
                    for j in random.sample(range(ln), n):
                        for i in random.sample(range(j+1,ln),min(1,ln-j-1)):
                            pairs[len(pairs)] = (instances[ph][j],instances[ph][i],0)
            n = int(math.ceil(math.sqrt(2.0*N/(len(instances)*(len(instances)-1)))))
            for j in range(len(instances)):
                for l in random.sample(range(len(instances[ks[j]])), min(n, len(instances[ks[j]]))):
                    for i in range(j+1,len(instances)):
                        for m in random.sample(range(len(instances[ks[i]])),min(n,len(instances[ks[i]]))):
                            pairs[len(pairs)] = (instances[ks[j]][l],instances[ks[i]][m],1)
    return pairs


def pkl2pairs(path, out_path, N = 1000000):
    pkl = read_pickles(path)
    instances = get_phone_instances(pkl)
    pairs = get_pairs(instances,N)
    if len(pairs) > 2*N:
        inds = [p for p in pairs]
        inds = random.sample(inds, 2*N)
    pairs = {i:pairs[i] for i in inds}
    pickle.dump(pairs, open(out_path,'wb'), protocol=2)


if __name__ == "__main__":
    if not os.path.exists('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/get_pairs.cmds','a') as f:
        f.write(' '.join(argv)+'\n')

    parser = argparse.ArgumentParser()
    parser.add_argument('-IN_PATH', nargs='+', required=True)
    parser.add_argument('-OUT_PATH', required=True)
    parser.add_argument('-N', required=False, default=100000, type=int)
    parser.add_argument('-SEED', required=False, default=100, type=int)
    args = parser.parse_args()
    random.seed(args.SEED)
    pkl2pairs(args.IN_PATH, args.OUT_PATH, args.N)
