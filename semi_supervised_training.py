import audiotesting
import scipy.io.wavfile
import numpy as np
import heapq
from numpy.fft import fft, ifft
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import os
import winsound
from keras.models import load_model
from keras.layers import Dense, Dropout
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn import mixture
from sklearn.neural_network import BernoulliRBM

############ import audio files and codewords


t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,sr = audiotesting.speakersets()
t = np.array([t0,t1,t2,t3,t4,t5,t6,t9,t8,t7])
pc= audiotesting.phones()	


###################output tensor

ln = [[[0 for _ in range(250)] ]for _ in range(250)]	
l=np.array(ln)	
for j in range(0, 250):	
	l[j,0,j] = 1


label =np.concatenate((l,l,l,l,l,l,l,l,l,l),axis=0)
label=np.array(label)



#############input tensor/mfcc/pad/normalize

k=0
mtrain = [[[] ]for _ in range(2500)]
for j in range(0, 10):
	for i in range(0, 250):		
		mtrain[k]= mfcc(t[j][i],sr,numcep=13)
		k=k+1
#mfclen

mfcmaxlen=0
mfclen=0
for j in range(0, 2500):
	mfclen=len(mtrain[j])
	if mfclen>mfcmaxlen:
		mfcmaxlen=mfclen


#pad

mptrain = [[[] ]for _ in range(2500)]
for j in range(0, 2500):
	mptrain[j]=np.pad(mtrain[j], [(0, mfcmaxlen-len(mtrain[j])),(0,0)], 'constant', constant_values=0)


mptrain=np.array(mptrain)


inputdata = [[ ]for _ in range(2500)]
for j in range(0, 2500):
	inputdata[j]=np.concatenate((mptrain[j,0:89,0],mptrain[j,0:89,1],mptrain[j,0:89,2],mptrain[j,0:89,3],mptrain[j,0:89,4],mptrain[j,0:89,5],mptrain[j,0:89,6],mptrain[j,0:89,7],mptrain[j,0:89,8],mptrain[j,0:89,9],mptrain[j,0:89,10],mptrain[j,0:89,11],mptrain[j,0:89,12]))


########## sampling

atrain=np.array(inputdata)
atrain=normalize(inputdata)

wrds=np.arange(2500)
wrd=np.arange(250)
spk=[0 for _ in range(250)]
k=0
for j in range(0,25):
	for i in range(0,10):
		spk[k]=250*i
		k=k+1


wrds2=wrds

trnwrds=spk+wrd
tstwrds=np.delete(wrds,trnwrds)

inputdim=623
atrain=atrain[:,0:inputdim]


labelpropsslerror=[0 for _ in range(50)]
feedforwardsslerror=[0 for _ in range(50)]


################## label propagation

#lptraindata=atrain

lptraindata=atrain
lplabels=[-1 for _ in range(2500)]
lplabels=np.array(lplabels)
lplabels[trnwrds]=trnwrds%250
#lplabels[0:250]=np.arange(250)


label_prop_model = LabelPropagation(alpha=1, gamma=20, kernel='rbf', max_iter=30, n_jobs=1,n_neighbors=7, tol=0.001)

#label_prop_model = LabelSpreading(alpha=1, gamma=20, kernel='rbf', max_iter=30, n_jobs=1,n_neighbors=7, tol=0.001)

label_prop_model.fit(lptraindata, lplabels)
lpclusters=label_prop_model.predict(lptraindata)[tstwrds]
lpclustersprob=label_prop_model.predict_proba(lptraindata)[tstwrds]
clustones=[0 for _ in range(2250)]
for j in range(0, 2250):
	if lpclusters[j]==tstwrds[j]%250:
		clustones[j]=1


clusterror=2250-sum(clustones)
labelpropsslerror[0]=clusterror/2250


########## feedforward

hidden_dim = 500


fftraindata=atrain[trnwrds]
fftestdata=atrain[tstwrds]
t_label=np.reshape(label,(2500,250))
fflabels=t_label[trnwrds]


input_flat = Input(shape=(inputdim,))
hidden_layer = Dense(hidden_dim, activation='relu')(input_flat)
#output_layer = Dense(250, activation='sigmoid')(hidden_layer)

output_layer = Dense(250, activation='softmax')(hidden_layer)
flatmodel = Model(input_flat, output_layer)
flatmodel.compile(optimizer='Adam', loss='categorical_crossentropy')
flatmodel.fit(fftraindata,fflabels,epochs=2000,batch_size=50,shuffle=True)
flat_out = flatmodel.predict(fftestdata)
flatword = np.argmax((flat_out),axis=1)
flaterror=0
for j in range(0, 2250):
	if flatword[j] == tstwrds[j]%250:
		flaterror = flaterror
	else:
		flaterror=flaterror+1

feedforwardsslerror[0]=flaterror/2250



#########self training


for k in range(1,50):
	ffconf=np.amax(flat_out,axis=1)
	ffconfwords=np.argsort(ffconf)
	lpconf=np.amax(lpclustersprob,axis=1)
	lpconfwords=np.argsort(lpconf)
	
	for i in range (2248,2250):
		fftraindata=np.append(fftraindata,atrain[lpconfwords[i]:lpconfwords[i]+1],axis=0)
		fflabels=np.append(fflabels,t_label[lpclusters[lpconfwords[i]]:lpclusters[lpconfwords[i]]+1],axis=0)
	
	
	for i in range (2248,2250):
		lplabels[ffconfwords[i]]=flatword[ffconfwords[i]]%250
	
	
	
	flatmodel.fit(fftraindata,fflabels,epochs=5*k,batch_size=50,shuffle=True)
	flat_out = flatmodel.predict(fftestdata)
	flatword = np.argmax((flat_out),axis=1)
	flaterror=0
	for j in range(0, 2250):
		if flatword[j] == tstwrds[j]%250:
			flaterror = flaterror
		else:
			flaterror=flaterror+1
	
	feedforwardsslerror[k]=flaterror/2250
	
	label_prop_model.fit(lptraindata, lplabels)
	lpclusters=label_prop_model.predict(lptraindata)[tstwrds]
	lpclustersprob=label_prop_model.predict_proba(lptraindata)[tstwrds]
	clustones=[0 for _ in range(2250)]
	for j in range(0, 2250):
		if lpclusters[j]==tstwrds[j]%250:
			clustones[j]=1
	
	
	clusterror=2250-sum(clustones)
	labelpropsslerror[k]=clusterror/2250
	

