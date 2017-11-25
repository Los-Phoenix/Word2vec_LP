#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Pan Yang (panyangnlp@gmail.com)
# Copyright 2017

from __future__ import print_function

import logging
import os
import sys
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 4:
        print("Useing: python train_word2vec_model.py input_text "
              "output_gensim_model output_word_vector")
        sys.exit(1)
    inp, outp1, outp2 = sys.argv[1:4]
    inp = "novelOne2.text.seg"
    model = Word2Vec(LineSentence(inp), size=100, window=10, min_count=3,
                     workers=multiprocessing.cpu_count())

    #model.save(outp1)
    #model.wv.save_word2vec_format(outp2)

    model.save("zh_novel_model_100")
    model.wv.save_word2vec_format("zh_novel_model_100_vec",
                                  "zh_novel_model_100_voc",
                                  binary=False)

    result = model.most_similar(u"早晨")
    #print(result)
    for e in result:
        print(e[0], e[1])
