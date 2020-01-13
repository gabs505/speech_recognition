from scipy.io import wavfile
import python_speech_features as psf
import numpy as np
from dtw import dtw
from numpy.linalg import norm
from scipy.interpolate import interp2d
import librosa
import math
from sklearn.cross_decomposition import CCA
from dtaidistance import dtw as dtw2





def framing(signal,fs):

    winlen=0.025
    winlen_s=int(winlen*fs) #winlen in samples

    overlap=25 #percent value
    winstep=int((1-overlap/100)*winlen_s) #gap between windows
    frames=[]


    for i in range(0, len(signal), winstep):
        frame=np.array([0]*(winlen_s-1))
        if len(signal)-i>winstep:
            frame=signal[i:i+winstep]

        #else:
            # frame[i:(len(signal)-i)]=signal[i:]

        frames.append(frame)

    return frames

def hanning(length):
    hann = []
    for i in range(length):
        x = 0.5 - 0.5 * math.cos((2 * math.pi * i) / (length - 1))
        hann.append(x)
    hann[0] = 0.0
    hann[-1] = 0.0
    return hann

def energy_log(frame):
    E=0
    for x in frame:
        E+=abs(x)**2
    if E!=0:
        E=20*np.log(E)
    # E = 20 * np.log(E)
    return E

def dct(frame,n_coeffs): #dct of a single frame
    N=40 #number of triangular bandpass filters
    E=frame
    # E=energy_log(frame)
    frame_coeffs=[]#vector of mfcc coeffs for frame
    c=0
    for m in range(n_coeffs):
        for k in range(N):
            m+=1
            k+=1 #because counting from zero
            c+=math.cos(m*(k-0.5)*math.pi/N)*E
        frame_coeffs.append(c)
    return frame_coeffs

def filter_banks(frames,fs,NFFT):
    nfilt=40
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / fs)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    return filter_banks

def mfcc(frames,fs):
    mfcc_coeffs=[]
    n=13 #number of MFCC coefficients
    for frame in frames:
        frame=frame*hanning(len(frame))
        # frame=frame*np.hanning(len(frame))
        N=512
        frame=np.fft.rfft(frame,N)
        frame=(abs(frame)**2)/N #power spectrum

        #applying mel filterbank
        frame=filter_banks(frame,fs,N)
        frame_coeffs=dct(frame,n)
        mfcc_coeffs.append(frame_coeffs)

    return mfcc_coeffs

def analysis(fs,signal): #returns array of mfcc coeffs for entire signal
    signal=signal[:,0]#taking one channel
    if len(signal) > max:
        signal = signal[0:max]
    signal= psf.sigproc.preemphasis(signal, coeff=0.95)
    frames=framing(signal,fs)
    word_coeffs=mfcc(frames,fs)
    # word_coeffs=mean_mfcc(word_coeffs)
    return word_coeffs

def mean_mfcc(coeffs):#vector of mean of mfccs from one frame
    means=np.array([0]*13)
    for i in range(len(coeffs)):
        for j in range(len(coeffs[0])):
            means[j]+=np.sum(coeffs[i][j])
    n=len(coeffs)
    means = [x / n for x in means]
    return means


max = 1500#max data length
labels = ['0','1','2','3','4','5','6','7','8','9']
dist = []
words = []
coeffs = []

#Creating word base
word_base={}
words=['zero','jeden','dwa','trzy','cztery','piec','szesc','siedem','osiem','dziewiec']
i=0
for l in labels:
    word_base[l]=words[i]
    i+=1

#Computing mfcc coeffs for word base


fs0, data0 = wavfile.read('./Baza/0.wav')
coeffs0 = np.array(analysis(fs0, data0))

fs1, data1 = wavfile.read('./Baza/1.wav')
coeffs1 = np.array(analysis(fs1, data1))

fs2, data2 = wavfile.read('./Baza/2.wav')
coeffs2 = np.array(analysis(fs2, data2))

fs3, data3 = wavfile.read('./Baza/3.wav')
coeffs3 = np.array(analysis(fs3, data3))

fs4, data4 = wavfile.read('./Baza/4.wav')
coeffs4 = np.array(analysis(fs4, data4))

fs5, data5 = wavfile.read('./Baza/5.wav')
coeffs5 = np.array(analysis(fs5, data5))

fs6, data6 = wavfile.read('./Baza/6.wav')
coeffs6 = np.array(analysis(fs6, data6))

fs7, data7 = wavfile.read('./Baza/7.wav')
coeffs7 = np.array(analysis(fs7, data7))

fs8, data8 = wavfile.read('./Baza/8.wav')
coeffs8 = np.array(analysis(fs8, data8))

fs9, data9 = wavfile.read('./Baza/9.wav')
coeffs9 = np.array(analysis(fs9, data9))


#---------------------------------------
counter=0
yes=0


for i in range(9):
    for j in range(5):
# for i in range(1):
#     for j in range(1):
        # ---input word analysis----
        file = './Cyfry/' + str(i) + str(j+2)+ '.wav'
        print('Number: ', i)
        # file='./Cyfry/' + str(0) + str(5)+ '.wav'
        # print(i)
        fs, data = wavfile.read(file)  # word to recognize
        if len(data) > max:
            data = data[0:max]
        input_mfcc = np.array(analysis(fs, data))
        # --------------------------

        #DTW
        #Comparing input word to word base
        d0, cost, acc_cost, path = dtw(coeffs0, input_mfcc, dist=lambda x, y: norm(x - y, ord=1))
        d1, cost, acc_cost, path = dtw(coeffs1, input_mfcc, dist=lambda x, y: norm(x - y, ord=1))
        d2, cost, acc_cost, path = dtw(coeffs2, input_mfcc, dist=lambda x, y: norm(x - y, ord=1))
        d3, cost, acc_cost, path = dtw(coeffs3, input_mfcc, dist=lambda x, y: norm(x - y, ord=1))
        d4, cost, acc_cost, path = dtw(coeffs4, input_mfcc, dist=lambda x, y: norm(x - y, ord=1))
        d5, cost, acc_cost, path = dtw(coeffs5, input_mfcc, dist=lambda x, y: norm(x - y, ord=1))
        d6, cost, acc_cost, path = dtw(coeffs6, input_mfcc, dist=lambda x, y: norm(x - y, ord=1))
        d7, cost, acc_cost, path = dtw(coeffs7, input_mfcc, dist=lambda x, y: norm(x - y, ord=1))
        d8, cost, acc_cost, path = dtw(coeffs8, input_mfcc, dist=lambda x, y: norm(x - y, ord=1))
        d9, cost, acc_cost, path = dtw(coeffs9, input_mfcc, dist=lambda x, y: norm(x - y, ord=1))
        # d0=dtw2.distance(coeffs0, input_mfcc)
        # d1=dtw2.distance(coeffs1, input_mfcc)
        # d2=dtw2.distance(coeffs2, input_mfcc)
        # d3=dtw2.distance(coeffs3, input_mfcc)
        # d4=dtw2.distance(coeffs4, input_mfcc)
        # d5=dtw2.distance(coeffs5, input_mfcc)
        # d6=dtw2.distance(coeffs6, input_mfcc)
        # d7=dtw2.distance(coeffs7, input_mfcc)
        # dist=[d0,d1,d2,d3,d4,d5,d6,d7]
        # print(dist)

        dist=[d0,d1,d2,d3,d4,d5,d6,d7,d8,d9]

        recognized_idx=dist.index(min(dist))#index of recognized word
        print('Recognized idx: ',recognized_idx)
        if recognized_idx==i:
            print("Word recognized correctly")
            yes+=1
        else:
            print("Error")
        counter+=1

        recognized_word = labels[recognized_idx]
        print('Recognized word is: ', word_base[recognized_word])



print('Recognition efficiency is: ', (yes/counter)*100,'[%]')






