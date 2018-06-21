import os
import cPickle
import numpy as np
from scipy.io.wavfile import read
from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")
import time
source   = "/home/tarek/dsp_project/Speaker-identification-using-GMMs/development_set/"

modelpath = "/home/tarek/dsp_project/Speaker-identification-using-GMMs/speaker_models/"
gmm_files = [os.path.join(modelpath,fname) for fname in
              os.listdir(modelpath) if fname.endswith('.gmm')]
models    = [cPickle.load(open(fname,'r')) for fname in gmm_files]
speakers   = [fname.split("\\")[-1].split(".gmm")[0] for fname
              in gmm_files]
sr, audio = read('/home/tarek/dsp_project/Speaker-identification-using-GMMs/get_speaker/test.wav')
vector = extract_features(audio, sr)

log_likelihood = np.zeros(len(models))

for i in range(len(models)):
    gmm = models[i]  # checking with each model one by one
    scores = np.array(gmm.score(vector))
    log_likelihood[i] = scores.sum()

winner = np.argmax(log_likelihood)
print "\tdetected as - ", speakers[winner]