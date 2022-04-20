#!/usr/bin/env python
# coding: utf-8

# In[187]:


class base:
    def plot_data(data,duration):
        sampling_freq = 160.0
        time = np.arange(0.0, duration, 1/sampling_freq)
        plt.plot(time,data)
    
    def segment_data(input_data, seg_time=30):
        # 30 seconds 
        segment_points = seg_time * 160 #sampling freq
        splited_data =np.asarray(np.split(input_data.flatten(), segment_points)).T
        
        return splited_data
    
    def form_data(input_data,attack_data):
        segment_time = 30 #window = 30seconds
        input_ = base.segment_data(input_data,segment_time)
        attack_ = base.segment_data(attack_data,segment_time)

        X = np.concatenate((input_,attack_))
        Y = np.concatenate((np.zeros(input_.shape[0]),np.ones(attack_.shape[0]))) #normal = 0, attack = 1
        
        return X,Y
        
    


# In[188]:


class feature:
    
    sampling_freq = 160.0
    
    def power_spectrum_plot(data, duration):
        #https://www.adamsmith.haus/python/answers/how-to-plot-a-power-spectrum-in-python
        
        #time_stop = 120sec
        time = np.arange(0.0, duration, 1/sampling_freq) #(start, stop, step)

        fourier_transform = np.fft.rfft(data)

        abs_fourier_transform = np.abs(fourier_transform)

        power_spectrum = np.square(abs_fourier_transform)

        frequency = np.linspace(0, sampling_freq/2, len(power_spectrum))
        
        plt.plot(frequency, power_spectrum)
        


# In[189]:


class preprocess:
    
    sampling_freq = 160.0
    
    def filter_band(data, duration):
        #high pass and low pass filter        
        sampling_freq = 160.0
        time = np.arange(0.0, duration, 1/sampling_freq)
        low_freq = 0.5 #0.1 Hz
        high_freq = 2.0 #60 Hz
        
        filter = signal.firwin(401, [low_freq, high_freq], pass_zero=False,fs=sampling_freq)
        
        filtered_signal = signal.convolve(data, filter, mode='same')
        
        plt.plot(time, filtered_signal)
    


# In[190]:


def main():
    from scipy.io import loadmat
    import numpy as np
    
    #Read data
    input_data = loadmat('Dataset1.mat') #dict_keys(['__header__', '__version__', '__globals__', 'Raw_Data', 'Sampling_Rate'])
    attack_data = loadmat('sampleAttack.mat')#dict_keys(['__header__', '__version__', '__globals__', 'attackVectors'])
    
    #loading data
    input_data = input_data['Raw_Data']
    attack_data = attack_data['attackVectors']
    
    #matrix of 106*3*19200 == > 106 subjects, 3 times of 2 min per subject, 
    #160 Hz sampling rate. (19200 = 120 s * 160 hz) 160 samples per second
    print("Input data shape: ", input_data.shape)

    #matrix of 106*3*19200 == > 6 attack types | 106 subjects | 3 times | 30 sec per subject, 
    #160 Hz sampling rate. (4800 = 30 s * 160 hz) 160 samples per second
    print("Attack data shape: ", attack_data.shape)
    
    #Combine all data 
    X,Y = base.form_data(input_data,attack_data)

    print(X.shape)
    print(Y.shape)
    
    
    #Divide data into train and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
    #Pass an int for reproducible output across multiple function calls
    
    
    
    
    


# In[191]:


if __name__ == "__main__":
    main()

