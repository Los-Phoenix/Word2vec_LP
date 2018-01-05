#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:LosPhoenix
# Based on: Pan Yang (panyangnlp@gmail.com)

from __future__ import print_function

import logging
import os
import sys
import multiprocessing
import time

reload(sys)
sys.setdefaultencoding('utf-8')

from gensim.models import Word2VecWC
from gensim.models.word2vecWC import LineSentence

folder_path = "../data/wiki_phrase2/"
ori_name = "wiki"
model_suffix = "_model"
vec_suffix = "_vec"
voc_suffix = "_voc"

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])

    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    CPUcount = multiprocessing.cpu_count()
    print(CPUcount)

    # inp = "../data/wikiDummy/wikiShort"
    inp = folder_path + ori_name
    lines = LineSentence(inp)
    print(lines.max_sentence_length)
    t = time.time()
    model = Word2VecWC(lines, size=60, window=10, min_count=3, workers=10, iter = 20)

    model.save(folder_path + ori_name + model_suffix)
    model.wv.save_word2vec_format(folder_path + ori_name + vec_suffix,
                                  folder_path + ori_name + voc_suffix,
                                  binary=False)
    logging.info("The time is %d", time.time() - t)
    result = model.most_similar(u"阿基米德")
    #print(result)
    for e in result:
        print(e[0], e[1])
