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

wrds=np.random.permutation(2500)
tstwrds=wrds[500:2500]
trnwrds=wrds[0:500]

atrain=np.array(inputdata)
atrain=normalize(inputdata)

#####################autoencoder



encoding_dim = 600 
input_mfc = Input(shape=(89*13,))
encoded = Dense(encoding_dim, activation='relu')(input_mfc)
decoded = Dense(89*13, activation='sigmoid')(encoded)
autoencoder = Model(input_mfc, decoded)
encoder = Model(input_mfc, encoded)
autoencoder.compile(optimizer='Rmsprop', loss='mse')
autoencoder.fit(atrain, atrain,epochs=50,batch_size=250,shuffle=True)

encoded_out = encoder.predict(atrain)


ltrain=encoded_out[trnwrds]
ltest=encoded_out[tstwrds]
t_label=np.reshape(label,(2500,250))
l_label=t_label[trnwrds]

####################feedforward model

hidden_dim = 500  
input_flat = Input(shape=(encoding_dim,))
hidden_layer = Dense(hidden_dim, activation='relu')(input_flat)
#output_layer = Dense(250, activation='sigmoid')(hidden_layer)

output_layer = Dense(250, activation='softmax')(hidden_layer)
flatmodel = Model(input_flat, output_layer)
flatmodel.compile(optimizer='Adam', loss='categorical_crossentropy')
flatmodel.fit(ltrain,l_label,epochs=1000,batch_size=250,shuffle=True)
#flatmodel.load_weights('D:/thesis/script/weights/14feb_feedforward_i1157h500o250_trn1250_e0000001.h5')

flat_out = flatmodel.predict(ltest)
flatword = np.argmax((flat_out),axis=1)
flaterror=0
for j in range(0, 2000):
	if flatword[j] == tstwrds[j]%250:
		flaterror = flaterror
	else:
		flaterror=flaterror+1

flaterror/2000

#flatmodel.fit(ltrain,l_label,epochs=50,batch_size=50,shuffle=True)
#flatmodel.save_weights('D:/thesis/script/weights/14feb_feedforward_i1157h500o250_trn1250_e0000001.h5')
#flatmodel.load_weights('D:/thesis/script/weights/14feb_feedforward_i1157h500o250_trn1250_e0000001.h5')


dgt=[0 for _ in range(1250)]
ffconferror=0
ffconfnumber=0
for j in range(0, 1250):
	if flat_out[j][flatword[j]]>0.99997:
		ffconfnumber=ffconfnumber+1
		dgt[j]=1
		if flatword[j]==tstwrds[j]%250:
			ffconferror = ffconferror
		else:
			ffconferror = ffconferror+1
	else:
		ffconferror = ffconferror

ffconferror
ffconfnumber

dgt=np.array(dgt)
newlabels=np.where(dgt==1)

for i in range(0,len(newlabels[0])):
	ltrain=np.append(ltrain,ltest[newlabels[0][i]:newlabels[0][i]+1,:],axis=0)

for i in range(0,len(newlabels[0])):
	l_label=np.append(l_label,t_label[flatword[newlabels[0][i]]:flatword[newlabels[0][i]]+1,:],axis=0)




#############label propagation

ptrain=atrain[wrds]
lplabels=[-1 for _ in range(2500)]
lplabels[0:1250]=trnwrds%250


label_prop_model = LabelPropagation(alpha=1, gamma=20, kernel='rbf', max_iter=30, n_jobs=1,n_neighbors=7, tol=0.001)
#label_prop_model = LabelSpreading(alpha=1, gamma=20, kernel='rbf', max_iter=30, n_jobs=1,n_neighbors=7, tol=0.001)

label_prop_model.fit(ptrain, lplabels)
lpclusters=label_prop_model.predict(ltest)
lpclustersprob=label_prop_model.predict_proba(ltest)
clusterror=0
for j in range(0, 250):
	if lpclusters[j]==tstwrds[j]%250:
		clusterror = clusterror
	else:
		clusterror = clusterror+1

clusterror/250



############ confidence measure

uclust=0

dgt1=[0 for _ in range(250)]
dgt2=[0 for _ in range(250)]


for j in range(0, 250):
	if lpclusters[j]==j:
		if lpclusters[j]==flatword[j]:
			uclust=uclust
		else:
			uclust=uclust+1


uclust


cconferror=0
cconfnumber=0
for j in range(0, 250):
	if lpclustersprob[j][lpclusters[j]]>.025:
		cconfnumber=cconfnumber+1
		dgt2[j]=1
		if lpclusters[j]==tstwrds[j]%250:
			cconferror = cconferror
		else:
			cconferror = cconferror+1
	else:
		cconferror = cconferror


cconferror
cconfnumber


cfconferror=0
cfconfnumber=0
for j in range(0, 250):
	if flat_out[j][flatword[j]]<1.0000065:
		if flat_out[j][flatword[j]]>0.00175:
			if lpclusters[j]==flatword[j]:
				cfconfnumber=cfconfnumber+1
				dgt1[j]=1
				if flatword[j]==j:
					cfconferror = cfconferror
				else:
					cfconferror = cfconferror+1
		else:
			cfconferror = cfconferror



cfconferror
cfconfnumber




dgt=np.array(dgt)
newlabels=np.where(dgt==1)
for i in range(0,len(newlabels[0])):
	lplabels[2250+newlabels[0][i]]=flatword[newlabels[0][i]]

for i in range(0,len(newlabels[0])):
	ltrain=np.append(ltrain,ltest[newlabels[0][i]:newlabels[0][i]+1,:],axis=0)

for i in range(0,len(newlabels[0])):
	l_label=np.append(l_label,l_label[flatword[newlabels[0][i]]:flatword[newlabels[0][i]]+1,:],axis=0)


