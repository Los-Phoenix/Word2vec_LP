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

from gensim.models import Word2VecWC
from gensim.models.word2vecWC import LineSentence

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])

    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    CPUcount = multiprocessing.cpu_count()
    print(CPUcount)

    # inp = "../data/wikiDummy/wikiShort"
    inp = "../data/novel/novelS"
    lines = LineSentence(inp)
    print(lines.max_sentence_length)
    t = time.time()
    model = Word2VecWC(lines, size=25, window=5, min_count=3, workers=10, sg = True, iter = 2)

    model.save("../data/novel/novelS_model")
    model.wv.save_word2vec_format("../data/novel/novelS_vec",
                                  "../data/novel/novelS_voc",
                                  binary=False)
    logging.info("The time is %d", time.time() - t)
    result = model.most_similar(u"阿基米德")
    #print(result)
    for e in result:
        print(e[0], e[1])
