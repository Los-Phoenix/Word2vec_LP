import cPickle as pickle

f1 = file('../../data/phraseClusterMean.pkl', 'rb')
samP = pickle.load(f1)
samP = pickle.load(f1)

print len(samP)