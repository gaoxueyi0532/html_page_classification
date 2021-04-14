# usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import os
import random
import json
import logging
import multiprocessing
import gensim
from time import sleep
from gensim.models import Word2Vec
from gensim import models
from gensim.models.word2vec import PathLineSentences

def train(path):
    # 日志信息输出
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # training
    model = Word2Vec(PathLineSentences("../data/tokens_v3.txt"), size=256, window=3, sg=1, min_count=1, iter=20)

    # save model
    model.save(path + "text_embed_v3.model")
    model.wv.save_word2vec_format(path + "binary_embed_v3.model", binary=True)

def test(path):
    #mpath = path + "binary_embed_v2.model"
    mpath = path + "binary_embed_v3.model"
    model = gensim.models.KeyedVectors.load_word2vec_format(mpath, binary=True)
    lst = model.most_similar(u'丰收')
    for item in lst:
        if sys.version.startswith('3'):
            print("%s : %s" % (str(item[0]), str(item[1])))
        else:
            print("%s : %s" % (str(item[0].encode('u8')), str(item[1])))

def main():
    path = "../model/"
    test(path)
    #train(path)

if __name__ == '__main__':
    main()
