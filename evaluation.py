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
import matplotlib.pyplot as plt

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

atrain=np.array(inputdata)
atrain=normalize(inputdata)
t_label=np.reshape(label,(2500,250))
wrd=np.arange(2500)
spk=[0 for _ in range(2500)]
k=0
for j in range(0,250):
	for i in range(0,10):
		spk[k]=250*i
		k=k+1

spk=np.array(spk)
wrds=(spk+wrd)%2500]


mf=9
inputdim=89*mf
dp=0.75
epc=500
hidden_dim = 100*hd
oact='softmax'
hact='relu'
bs=250
opt='Adam'
lss='categorical_crossentropy'
z=9


trnwrds=wrds[0:z*250]
tstwrds=wrds[z*250:2500]

ptrain=atrain[wrds,0:inputdim]
plabels=t_label[wrds]


ltrain=atrain[trnwrds,0:inputdim]
ltest=atrain[tstwrds,0:inputdim]
l_label=t_label[trnwrds]

####################feedforward model

 
input_flat = Input(shape=(inputdim,))
hidden_layer = Dense(hidden_dim, activation=hact)(input_flat)
dropout_layer = Dropout(dp)(hidden_layer)
output_layer = Dense(250, activation=oact)(dropout_layer)
flatmodel = Model(input_flat, output_layer)
flatmodel.compile(optimizer=opt, loss=lss)
hist=flatmodel.fit(ptrain,plabels,epochs=epc,batch_size=bs,shuffle=True,validation_split=0.1)
flat_out = flatmodel.predict(ltest)
flatword = np.argmax((flat_out),axis=1)
flaterror=0
for j in range(0, 250):
	if flatword[j] == tstwrds[j]%250:
		flaterror = flaterror
	else:
		flaterror=flaterror+1

percenterror=flaterror/2.5
mtitle='model loss_error '+ str(percenterror)+'% dropout '+str(dp)+' hid '+hidden_dim
plt.figure(hd)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title(mtitle)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


plt.show()


