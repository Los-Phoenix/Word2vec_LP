# coding: utf-8
#!/usr/bin/env python

#This function gets wiki titles and saves it somewhere

from __future__ import print_function

import logging
import os.path
import six
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from LPWiki import WikiCorpus

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if False:
        print("Using: python process_wiki.py enwiki.xxx.xml.bz2 wiki.en.text")
        sys.exit(1)

    inp = '../data/zhwiki.xml.bz2'
    outp = '../data/zhwiki.txt'
    space = " "
    i = 0

    output = open(outp, 'w')
    wiki = WikiCorpus(inp, lemmatize=False, dictionary={}, )

    for text in wiki.get_texts():
        if six.PY3:
            output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
        #   ###another method###
        #    output.write(
        #            space.join(map(lambda x:x.decode("utf-8"), text)) + '\n')
        else:
            output.write(text[1][1] + "\n")
        i = i + 1
        if (i % 10000 == 0):
            logger.info("Saved " + str(i) + " articles")
        # if i == 1000:
        #     break

    output.close()
    logger.info("Finished Saved " + str(i) + " articles")
