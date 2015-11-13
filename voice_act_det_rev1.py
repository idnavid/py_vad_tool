#! /usr/bin/python

# Voice Activity Detection (VAD) tool.
# use the vad_help() function for instructions.
# Navid Shokouhi December 2012.

# The revised version includes addition features Voice Activity Detection. Including:
# SFM (Spectral Flatness Measure)
import wave 
import struct
import time 
import math
import sys
#sys.path.append('/usr/share/pyshared/scipy')
import scipy as sc
from scipy import cluster
import numpy as np
from numpy.fft import fft, rfft
import os
#import pylab

##Function definitions:
def vad_help():
	"""Voice Activity Detection (VAD) tool.
	
	Navid Shokouhi July 2012.
	"""
	print 10*'-','Voice Activity Detection (VAD) tool',10*'-'
	print 3*' ','Usage:'
	print 3*' ','python voice_act_dect.py <OPTIONS> <INPUT> <OUTPUT>'+'\n'
	print 3*' ','OPTIONS:'
	print 3*' ','-e              only use log-energy for vad (default: Off)'+'\n'
	print 3*' ','-ez             use zero-crossing-rate and log-energy (default: Off)'+'\n'
	print 3*' ','-es             use spectral flatness measure and log-energy (default: Off)'+'\n'
	print 3*' ','-ezs            use zero-crossing-rate, spectral flatness measure and log-energy (default: On)'+'\n'
	print 3*' ','-l              set output to be labels for each file (0: nonspeech, 1: voiced). Default: On'+'\n'
	print 3*' ','-i              set output to be indexes for voiced frames. Default: Off'+'\n'

def read_wav(filename):
	"""read file in wav format and return output in list format. One value per sample'
	Input: (filename) string
		filename	string(including location). Note: '~' is not interpreted as $HOME
	Output: (s, fs) tuple
		s			list of samples. Each sample is of type float and s is ranged in [-1,1] 
		fs			sampling frequency(aka frame rate, python terminology)"""
	wid=wave.open(filename,'r')
	if not (wid.getnchannels() == 1):
		print "Function only accepts single-channel wav files."
		exit 
	fs=wid.getframerate() # framerate is the sampling rate
	n_samples=wid.getnframes() # samples is the same as frames
	s_raw=wid.readframes(n_samples)
	s1=struct.unpack_from("%dh"%n_samples,s_raw)
	# normalize samples to [-1,1]
	s_max=max(abs(min(s1)),abs(max(s1)))
	s=[]
	for i in s1:
		s.append(i*1.0/s_max)
	return list(s), fs

def enframe(s_list, win, inc):
	"""enframe: Break input list into frames of length win. The frames
	are overlapped by the amount of win-inc.
	The name was inspired by the enframe MATLAB function provided in voicebox.
	Inputs: (s_list, win, inc)
		s_list: 	input list
		win: 		frame length
		inc: 		increment of next frame with respect to the start point of the previous frame
	Output:
		frames: 	list of lists. The inner lists are the frames
	"""
	win1=int(win)
	inc1=int(inc)
	s_temp=s_list[:]#prevent changing the original list
	n_samples=len(s_temp)
	n_frames=n_samples/inc1
	#zeropad in case of mismatch (i.e. ((n_frames*inc1+win1)-n_samples)~=0).
	for z_ind in range(((n_frames*inc1+win1)-n_samples)):
		s_temp.append(0.0)
	frames=[[]]*n_frames
	for i in range(n_frames):
		frames[i]=s_temp[i*inc1:i*inc1+win1]
	return frames

def geomean(val_array):
	"""Geometric mean function to be used in computing the Spectral Flatness measure of a frame (GM/AM)"""
	p=0
	for i in val_array:
		p+=np.log(i)
	return math.exp((1.0/len(val_array))*p)

def main_vad(frames,e=1,z=1,sf=1):
	"""main_vad: label frames as voiced(1) and unvoiced(0) and store 
	labels in output list (vad).
	The features used in the vad are log-energy and zero-crossing-rate.
	Inputs: (frames, e=0)
		frames: 	list of lists. The inner list are the samples (type float) 
		e:	 		flag indicating whether to use log-energy if e==1 use log-energy. 
		z:			flag indicating whether to use zero crossing rate or not. if z==1, use zero crossing rate.
		sf:			flag indicating whether to use spectral flatness measure or not. if sf==1, use spectral flatness measure. 
		NOTE: By default all three flags (e, z, and s are On)
	Output:
		vad: 		list of 0's and 1's and same length of frames. 1 indicates frame is voiced.
					Elements are of type int.
	"""
	#Compute log energy of frames
	edB=[]
	for i in frames:
		temp=i[:]
		e_tmp=math.fsum(np.array(temp)**2)
		edB.append(10*math.log10(e_tmp))
		
	#Alternative method
	###EPSILON=10**(-12)	
	###for i in f:
		###e_tmp=0
		###for j in tuple(i):
			###e_tmp+=j**2
		###edB.append(10*math.log10(e_tmp+EPSILON))

	#Compute Zero Crossing Rate of frames
	zcr=[]
	for i in frames:
		z_ind=0
		for j in range(len(i)-1):
			if (i[j]>0 and i[j+1]<0) or (i[j]==0):
				z_ind=z_ind+1
		zcr.append(z_ind*1.0/len(i))
		
	#Compute Spectral Flactness Measure of frames
	sfm=[]
	EPSILON=10**(-12)
	fft_size=512
	for i in frames:
		frame_windowed=np.array(i)*np.hanning(len(i))# Multiply frame by hanning window
		spect=rfft(frame_windowed,fft_size)
		mag_spect=np.absolute(spect)# Compute magnitude spectrum
		temp_s=(geomean(mag_spect+EPSILON))/np.mean(mag_spect)
		sfm.append(temp_s)
		# The following commented lines were used for debugging
		#print s
		#pylab.plot(mag_spect)
		#pylab.show()
		#time.sleep(1)
	ZCR=np.array(zcr).reshape(len(zcr),1)
	EDB=np.array(edB).reshape(len(edB),1)
	SFM=np.array(sfm).reshape(len(sfm),1)
	feat=np.concatenate((ZCR,EDB,SFM),axis=1)
	print e, z, sf
	#Find out which cluster is voiced:
	vad=[]
	if e==1 and z==1 and sf==1:
		[cents,labels]=cluster.vq.kmeans2(feat,2)
		print cents
		label_list=labels.flatten().tolist()
		#check if mean energy/zcr of cluster 0 is greater/smaller than that of cluster 1		
		if (cents[0,0]<cents[1,0]) and (cents[0,1]>cents[1,1]) and (cents[0,2]<cents[1,2]):
			#cluster 0 is voiced:
			for i in label_list:
				if i==0:
					vad.append(1)
				else:
					vad.append(0)
		elif (cents[0,0]>cents[1,0]) and (cents[0,1]<cents[1,1]) and (cents[0,2]>cents[1,2]):
			#cluster 1 is voiced:
			for i in label_list:
				if i==1:
					vad.append(1)
				else:
					vad.append(0)
		else:
			#Must only use energy
			print 'Warning >>> data is too noisy, must only use energy; e is turned On'
			[cents1,labels1]=cluster.vq.kmeans2(EDB,2)
			label_list1=labels1.flatten().tolist()
			if cents1[0]>cents1[1]:
				#cluster 0 is voiced:
				for i in label_list1:
					if i==0:
						vad.append(1)
					else:
						vad.append(0)
			else:
				#cluster 1 is voiced:
				for i in label_list1:
					if i==1:
						vad.append(1)
					else:
						vad.append(0)
	elif e==1 and z==1 and sf==0:
		# ez flag is on.
		feat1=np.concatenate((ZCR,EDB),axis=1)# Only use log-energy and zero crossing rate  
		[cents,labels]=cluster.vq.kmeans2(feat1,2)
		print cents
		label_list=labels.flatten().tolist()
		if (cents[0,0]<cents[1,0]) and (cents[0,1]>cents[1,1]):
			#cluster 0 is voiced:
			for i in label_list:
				if i==0:
					vad.append(1)
				else:
					vad.append(0)
		elif (cents[0,0]>cents[1,0]) and (cents[0,1]<cents[1,1]):
			#cluster 1 is voiced:
			for i in label_list:
				if i==1:
					vad.append(1)
				else:
					vad.append(0)
		else:
			#Must only use energy
			print 'Warning >>> data is too noisy, must only use energy; e is turned On'
			[cents1,labels1]=cluster.vq.kmeans2(EDB,2)
			label_list1=labels1.flatten().tolist()
			if cents1[0]>cents1[1]:
				#cluster 0 is voiced:
				for i in label_list1:
					if i==0:
						vad.append(1)
					else:
						vad.append(0)
			else:
				#cluster 1 is voiced:
				for i in label_list1:
					if i==1:
						vad.append(1)
					else:
						vad.append(0)
	elif e==1 and z==0 and sf==1:
		# es flag is on.
		feat1=np.concatenate((SFM,EDB),axis=1)# Only use log-energy and zero crossing rate  
		[cents,labels]=cluster.vq.kmeans2(feat1,2)
		print cents
		label_list=labels.flatten().tolist()
		if (cents[0,0]<cents[1,0]) and (cents[0,1]>cents[1,1]):
			#cluster 0 is voiced:
			for i in label_list:
				if i==0:
					vad.append(1)
				else:
					vad.append(0)
		elif (cents[0,0]>cents[1,0]) and (cents[0,1]<cents[1,1]):
			#cluster 1 is voiced:
			for i in label_list:
				if i==1:
					vad.append(1)
				else:
					vad.append(0)
		else:
			#Must only use energy
			print 'Warning >>> data is too noisy, must only use energy; e is turned On'
			[cents1,labels1]=cluster.vq.kmeans2(EDB,2)
			label_list1=labels1.flatten().tolist()
			if cents1[0]>cents1[1]:
				#cluster 0 is voiced:
				for i in label_list1:
					if i==0:
						vad.append(1)
					else:
						vad.append(0)
			else:
				#cluster 1 is voiced:
				for i in label_list1:
					if i==1:
						vad.append(1)
					else:
						vad.append(0)
	elif e==1 and z==0 and sf==0:
		# e flag is on:
		[cents1,labels1]=cluster.vq.kmeans2(EDB,2)
		print cents1
		label_list1=labels1.flatten().tolist()
		if cents1[0]>cents1[1]:
			#cluster 0 is voiced:
			for i in label_list1:
				if i==0:
					vad.append(1)
				else:
					vad.append(0)
		else:
			#cluster 1 is voiced:
			for i in label_list1:
				if i==1:
					vad.append(1)
				else:
					vad.append(0)
	return vad

#End of function definitions.

##Read inputs:
inputs=sys.argv
#print inputs

#predefine options
e=1
z=1
sf=1
l=1
idx=0
if ((len(inputs)<3) or (len(inputs)>5)):
	print "\n  Error >>> Wrong number of inputs\n"
	vad_help()
	raise ValueError
else:
	for ind in range(len(inputs)):
		if inputs[ind]=='-e':
			e=1
			z=0
			sf=0
		if inputs[ind]=='-ez' or inputs[ind]=='-ze':
			e=1
			z=1
			sf=0
		if inputs[ind]=='-es' or inputs[ind]=='-se':
			e=1
			z=0
			sf=1
		if inputs[ind]=='-ezs' or inputs[ind]=='-esz':
			e=1
			z=1
			sf=1
		if inputs[ind]=='-i':
			idx=1
			l=0
if e==0:
	print '\n  Error >>> e must be on since log-energy must always be one of the features.\n'
	raise ValueError

if idx==l:
	print '\n  Error >>> i and l cannot be the same\n'
	raise ValueError
	
filename=inputs[-2]
try:
	read_wav(filename)
except:
	print "\n  Error >>> Input file ""'%s'"" is not a valid wavefile\n"% filename
	raise ValueError

output=inputs[-1]

##End of Read inputs.


s,fs=read_wav(filename)
f=enframe(s, int(0.025*fs), int(0.01*fs))
#print e, z, sf
if z==1 and sf==1:
	#print "I'm in z=1 sf=1"
	vad=main_vad(f,e=1,z=1, sf=1)
elif z==1 and sf==0:
	#print "I'm in z=1 sf=0"
	vad=main_vad(f,e=1,z=1,sf=0)
elif z==0 and sf==1:
	#print "I'm in z=0 sf=1"
	vad=main_vad(f,e=1,z=0,sf=1)	
elif z==0 and sf==0:
	#print "I'm in z=0 sf=0"
	vad=main_vad(f,e=1,z=0,sf=0)	
else:
	print "\n   Error >>> The combination of features you chose is incorrect.\n"
	raise ValueError

##Write output:
# For convenience there are two ways of storing the output:
# 1. 0's and 1's per frame. l==1 (-l)
fout=open(output,'w')
if l==1:
	for i in vad:
		fout.write(str(i)+'\n')
		
# 2. index of voiced frames. idx==1 (-i)
elif idx==1:
	cnt=0
	for i in vad:
		if i==1:
			fout.write(str(cnt)+'\n')
			cnt+=1
		else:
			cnt+=1
			

