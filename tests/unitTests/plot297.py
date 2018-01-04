#coding:utf-8
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys, getopt
import gensim
import numpy as np
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'

reload(sys)
sys.setdefaultencoding('utf-8')

def plot_with_labels(low_dim_embs, labels, filename = 'tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18,18))
    for i, label in enumerate(labels):
        x,y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy = (x,y),
                     xytext=(5,2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom'
                     )
        plt.savefig(filename)

tsne = TSNE(perplexity = 30, n_components=2, init='pca', n_iter=5000)
plot_only = 100


fSimDict297 = list(open("../../data/297.txt"))

model0 = gensim.models.Word2VecWC.load("../../data/novel/novelS_model")
model500 = gensim.models.Word2VecWC.load("../../data/novel/novelS_model_500")
model1000 = gensim.models.Word2VecWC.load("../../data/novel/novelS_model_1000")

listWords = list()
list297 = list()
listsim0 = list()
listsim500 = list()
listsim1000 = list()

wordSet = set()
final_embedding = list()
reverse_dictionary = list()
for i in fSimDict297:

    w0, w1, sim = i.decode().split('\t')

    if w0 in model0 and w1 in model0:
        if not w0 in wordSet:
            reverse_dictionary.append(w0)
            final_embedding.append(model0[w0])
        if not w1 in wordSet:
            reverse_dictionary.append(w1)
            final_embedding.append(model0[w1])

final_embedding = np.asarray(final_embedding)

low_dim_embs = tsne.fit_transform(final_embedding[:5, :])
labels = [reverse_dictionary[i] for i in range(len(low_dim_embs))]
print labels
plot_with_labels(low_dim_embs, labels)

