import pickle

def savepkl(fname, data):
	with open(fname, 'wb') as fp:
		pickle.dump(data, fp)

def loadpkl(fname):
	with open(fname, 'rb') as fp:
		data = pickle.load(fp)
	return data