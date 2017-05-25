#! /usr/bin/python

# Voice Activity Detection (VAD) tool.
# use the vad_help() function for instructions.
# Navid Shokouhi December 2012.

# Updated: May 2017 for Speaker Recognition collaboration.

# The revised version includes addition features Voice Activity Detection. Including:
# SFM (Spectral Flatness Measure)
from audio_tools import *
import numpy as np
#import pylab

##Function definitions:
def vad_help():
	"""Voice Activity Detection (VAD) tool.
	
	Navid Shokouhi July 2012.
	"""


#### Energy tools
def zero_mean(xframes):
    """
        remove mean of framed signal
        return zero-mean frames.
        """
    m = np.mean(xframes,axis=1)
    xframes = xframes - np.tile(m,(xframes.shape[1],1)).T
    return xframes

def geo_mean(spect_frames):
    """Geometric mean of absolute spectrum
        """
    return 0



def compute_nrg(xframes):
    return np.diagonal(np.dot(xframes,xframes.T))

def compute_log_nrg(xframes):
    return np.log(compute_nrg(xframes+1e-4))

def power_spectrum(xframes):
    """
        x: input signal, each row is one frame
        """
    X = np.fft.fft(xframes,axis=1)
    X = np.abs(X[:,:X.shape[1]/2])**2
    return np.sqrt(X)

def compute_sfm(xframes):
    X = power_spectrum(xframes)
    return 0



def main_vad(frames):
    frames = zero_mean(frames)
    
    # Compute per frame energies:
    frame_nrgs = compute_nrg(frames)

if __name__=='__main__':
    test_file='/Users/navidshokouhi/Software_dir/subspace_speechenhancement/data/sa1-falr0_noisy.wav'
    fs,s = read_wav(test_file)
    s_frames = enframe(s,400,160) # rows: frame index, cols: each frame
    S = power_spectrum(s_frames)
    print S.shape


