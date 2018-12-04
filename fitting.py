import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatProgress
from scipy.signal import hilbert
import scipy.fftpack as fft
from scipy.signal import butter, lfilter
import scipy.stats as stats

#Kuramoto code
def solve_Kuramoto(t, theta0, w, C, G, b, beta):
    """
    Implementation of Kuramoto coupling model
    Usage example:
    >>> 
    >>> 
    Prams:

    """
    
    theta = np.empty((len(theta0),len(t))) 
    
    dt = t[1]-t[0]
    
    theta[:,0] = theta0
    
    dsig = np.sqrt(dt) * beta

    #num oscillators
    n = len(theta0)
    
    #temporal loop
    for t_step in range(1,len(t)):
        
        Diff = np.sin(np.tile(theta[:,t_step-1],(n,1))-np.transpose(np.tile(theta[:,t_step-1],(n,1))))
        
        theta[:,t_step] = theta[:,t_step-1] + dt*(w- b*np.sin(theta[:,t_step-1])+ G*(np.sum(C*Diff,axis=1))) + \
                                            + dsig*np.random.normal(0,dsig,n)
    

    #Order param calculation   
    stot = np.zeros(t.shape)
    ctot = np.zeros(t.shape)

    for i in range(n):
       stot = stot + np.sin(theta[i,:])
       ctot = ctot + np.cos(theta[i,:])
    stot = stot/n
    ctot = ctot/n

    R = np.sqrt(ctot**2+stot**2)

    return theta, R

def get_R(theta):
    #Order param calculation   
    stot = np.zeros(t.shape)
    ctot = np.zeros(t.shape)

    for i in range(n):
       stot = stot + np.sin(theta[i,:])
       ctot = ctot + np.cos(theta[i,:])
    stot = stot/n
    ctot = ctot/n

    R = np.sqrt(ctot**2+stot**2)

def filter_bandpass(data, lowcut, highcut, fs, order=5):
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    
    return y

def filtered_data(data, lowcut, highcut, fs, order=5):

    N = data.shape[0]
    
    ts_filtered = np.empty_like(data)
    
    for i in range(N):
        ts_filtered[i,:] = filter_bandpass(data[i,:],lowcut,highcut,fs)
    return ts_filtered

def extract_freq(data,t,plot=False):
    
    N = len(data)

    yf = fft.fft(data)
    xf = np.linspace(0.0, 1.0/(2.0*(t[1]-t[0])), N//2)
    power1 = 2.0/N*np.abs(yf[:N//2])

    if plot:
        plt.plot(xf, power1, )
        plt.xlabel('Freqs [Hz]')
        plt.ylabel('Power')
        plt.show()

    max_freq = xf[power1.argmax()]
    #print("Maximum power: ",max_freq,"Hz")
    return max_freq
    
def w_distribution(data,t):
    
    N = data.shape[0]
    
    w_dist = np.empty(N)
    
    for i in range(N):
        w_dist[i] = 2*np.pi*extract_freq(data[i,:],t)
        
    return w_dist

def bold_to_phase(signal):
    phase = np.empty_like(signal)
    amplitude = np.empty_like(signal)
    
    n = signal.shape[0]
    
    for i in range(n):
    
        analytic_signal = hilbert(signal[i,:])
        
        amplitude[i,:] = np.abs(analytic_signal)
        phase[i,:] = np.unwrap(np.angle(analytic_signal))
    
    #return amplitude,phase
    
    return phase

def phases_restrict(theta):
    return np.mod(theta+2*np.pi, 2*np.pi)-np.pi

def phase_diff(data):
    
    N = data.shape[0]
    Nt = data.shape[1]
    
    phase_diff_lst = []
    
    for i in range(N):
        for j in range(i+1,N):
            phase_diff_lst.append(data[i,:] - data[j,:])
    
    phase_diff_matrix = np.array(phase_diff_lst)
    phase_diff_matrix[phase_diff_matrix > np.pi] = phase_diff_matrix[phase_diff_matrix > np.pi] - 2*np.pi
    phase_diff_matrix[phase_diff_matrix < -np.pi] = phase_diff_matrix[phase_diff_matrix < -np.pi] + 2*np.pi
    
    return phase_diff_matrix

def div_kl(dist1,dist2):
    res = stats.entropy(dist1,dist2)
    return res


def plot_circle(theta, t, R=None):
    import time
    from IPython import display

    fig, ax = plt.subplots(1,2,figsize=(12,6))
    plt.figure(1, figsize=(14,2))
    fig.tight_layout()

    for j in range(len(t)):
        s1 = ax[0].scatter(np.cos(theta[:,j]),np.sin(theta[:,j]),c='k')
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)
        s2 = ax[1].plot(t[0:j],R[0:j],color='b')
        ax[1].set_xlim(0,t[-1])
        ax[1].set_ylim(0,1)
        ax[1].set_ylabel("$R$")
        ax[1].set_xlabel("$t$")
        display.clear_output(wait=True)
        display.display(plt.gcf())
        time.sleep(0.005)
        s1.remove()  

    print("Display over!")