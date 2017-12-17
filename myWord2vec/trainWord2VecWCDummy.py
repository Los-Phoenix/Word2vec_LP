#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:LosPhoenix
# Based on: Pan Yang (panyangnlp@gmail.com)

from __future__ import print_function

import logging
import os
import sys
import multiprocessing

from Word2VecWC import Word2VecWC

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])

    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    CPUcount = multiprocessing.cpu_count()
    print(CPUcount)

    inp = "../data/novel2/novelOne2.text.seg"
    lines = LineSentence(inp)
    print(lines.max_sentence_length)

    model = Word2VecWC(lines, size=50, window=5, min_count=20,workers=CPUcount)
    #model.save(outp1)
    #model.wv.save_word2vec_format(outp2)

    model.save("../data/novel2/Dummy_model")
    model.wv.save_word2vec_format("../data/novel2/Dummy_model_vec",
                                  "../data/novel2/Dummy_model_voc",
                                  binary=False)

    result = model.most_similar(u"早晨")
    #print(result)
    for e in result:
        print(e[0], e[1])
