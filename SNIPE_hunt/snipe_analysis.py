'''
THIS FILE CONTAINS FUNCTIONS USED IN THE ANALYSIS OF SNIPE HUNT DATA NOTEBOOKS
Assembled from new previous code beginning in July 2024
'''

# LIBRARIES -----------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

import h5py
import os


# EXTRACTION/ACQUIRE ---------------------------------------------------------------------------------------------------
def pull_record(filename):
    '''
    Pulls data and timestamps from a SNIPE Magnetometer .h5 file.  Cleans the timestamps and puts them into a 1D array.
    Puts the magnetometer reading data into a 2D array, where each column represents a second and each row one of the
    2597 individual samples for that seconds in chronological order.
    The function also captures the timestamp associated with the filename for possible use.
    
    Parameters:
    - filename (string): .h5 data file.  Ensure the full path is correct!
    
    Returns:
    - file_timestamp (datetime): Timestamp for file (minute worth of data)
    - timestamps (datetime array): Timestamps associated with data 'seconds'
    - data (2D float array): Data associated with timestamps
    
    Note: The function Transposes the data array to make it easier for manipulation.
    '''
    index = np.array(h5py.File(filename)['timestamps'])
    data = np.array(h5py.File(filename)['data']).T # See note above
    
    timestamps = []
    
    try:
        # In order to accomodate different file/directory combinations, the next line identifies the start index required
        start_index = filename.find('2024')
        # This next line of code extracts the minute record's timestamp, in datetime format
        file_timestamp = datetime.strptime(filename[start_index:-3], '%Y-%m-%d_%H-%M-%S-%f')
    except:
        print('Filename not in proper format')
        file_timestamp = timestamps[0]
    
    for i in index:
        # Decode the byte string to a regular string
        decoded_string = i.decode('utf-8')

        # Convert the string to datetime format (the try except was just put in to address bad data in file)
        try:
            d = datetime.strptime(decoded_string, '%Y-%m-%d %H:%M:%S.%f')
            timestamps.append(d)
        except:
            decoded = decoded_string[:19] + '.' + decoded_string[19:]
            d = datetime.strptime(decoded, '%Y-%m-%d %H:%M:%S.%f')
            timestamps.append(d)

    return file_timestamp, timestamps, data

def interpolate_timestamps(timestamps, sample_rate = 2597):
    '''
    Create an np array with the interpolated timestamp for EVERY sample within the minute
    '''
    # Initialize Data Holder to return
    interpolated_timestamps = []

    # Since the timestamps are for the end of the second, we do all but the first second first
    interval_sum = []
    for i in range(1,len(timestamps)):
        interval = (timestamps[i] - timestamps[i-1])/sample_rate
        interval_sum.append(interval)
        for j in range(sample_rate):
                interpolated_timestamps.append(timestamps[i-1] + (interval * j))
    # Need to manually add in last timestamp
    interpolated_timestamps.append(timestamps[-1])
    # Add in the first second using the average interval from the previous
    average_interval = sum(interval_sum, timedelta())/len(interval_sum)
    for k in range(1,sample_rate):
        interpolated_timestamps.insert(0,timestamps[0] - (average_interval * k))
        
    return np.array(interpolated_timestamps)

def flatten_data(data):
    '''
    Takes in a 2D array of time data (seconds, samples) and converts it to one long array (samples).
    
    Ensure the data is properly transposed!
    '''
    
    return data.flatten()

def extract_complete_record(filename):
    '''
    Returns an np.array with all timestamps indexed to another np.array with all the data readings at those timestamps.
    '''
    file_timestamp, timestamps, data = pull_record(filename)
    
    timestamps = interpolate_timestamps(timestamps, sample_rate = 2597)
    data = flatten_data(data)
    
    return pd.Series(data, index = timestamps)


# DATA CONSOLIDATION -------------------------------------------------------------------
def get_minute_stats(timestamps, data):
    '''
    Downsamples record to the minute, returning the mean, std and max difference
    from the mean magnetometer reading within that minute.
    '''
    time = timestamps[0]
    
    avg = data.mean()
    mx = max(data)
    mn = min(data)
    stddev = np.std(data)
    max_diff = max(abs(mx-avg),abs(mn-avg))
    
    output = {'time':time, 'mean':avg, 'std':stddev, 'max_diff':max_diff}
    
    return output

def by_minute_descriptive_stats(directory, store = False):
    '''
    Takes in a directory of records and returns the descriptive statistic for that minute
    
    '''
    # List to store DataFrames
    minutes = []

    # To ensure proper ordering of data
    file_list = [f for f in os.listdir(directory) if f.endswith(".h5")]
    file_list.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)))

    # For tracking progress
    counter = 1

    # Iterate over each file in the directory
    for filename in file_list:
        if filename.endswith(".h5"):
            file_path = os.path.join(directory, filename)

            file_timestamp, timestamps, data = pull_record(file_path)
            
            data = flatten_data(data)

            minute = get_minute_stats(timestamps, data)

            # Append the DataFrame to the list
            minutes.append(minute)

            # Update counter for progress tracking
            counter += 1
            print(f'Completed {counter} of {len(file_list) + 1} files', end = '\r')

    # OPTIONAL: save to a single csv
    if store == True:
        minutes.to_csv(f'minute_data_{directory[-6:-1]}.csv', index=False)
        
    return pd.DataFrame(minutes).set_index('time')

def get_second_stats(timestamps, data):
    '''
    Downsamples record to the minute, returning the mean, std and max difference
    from the mean magnetometer reading within that minute.
    '''
    
    output = []

    for i, second in enumerate(data):
        time = timestamps[i]
        avg = second.mean()
        mx = max(second)
        mn = min(second)
        stddev = np.std(second)
        max_diff = max(abs(mx-avg),abs(mn-avg))
        output.append({'time':time, 'mean':avg, 'std':stddev, 'max_diff':max_diff})
    
    return output

def by_second_descriptive_stats(directory, store = False):
    '''
    Takes in a directory of records and returns the descriptive statistics for the aggregated seconds,    
    '''
    # List to store DataFrames
    seconds = []

    # To ensure proper ordering of data
    file_list = [f for f in os.listdir(directory) if f.endswith(".h5")]
    file_list.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)))

    # For tracking progress
    counter = 1

    # Iterate over each file in the directory
    for filename in file_list:
        if filename.endswith(".h5"):
            file_path = os.path.join(directory, filename)

            file_timestamp, timestamps, data = pull_record(file_path)

            seconds_stats = get_second_stats(timestamps, data)

            # Append the DataFrame to the list
            seconds.extend(seconds_stats)

            # Update counter for progress tracking
            counter += 1
            print(f'Completed {counter} of {len(file_list) + 1} files', end = '\r')

    # OPTIONAL: save to a single csv
    if store == True:
        seconds.to_csv(f'minute_data_{directory[-6:-1]}.csv', index=False)
        
    return pd.DataFrame(seconds).set_index('time')
# --------------------------------------------------------------------------------------------------


# FUNCTIONS - OUTLIERS -----------------------------------------------------------------------------
def outlier_filter(df_m, stddev_cutoff):
    '''
    Takes in a 
    
    '''
    std_mean = df_m.describe().loc['mean','std'] # df_m.describe() gives the descriptive statistics.  
    std_std = df_m.describe().loc['std','std']

    max_diff_mean = df_m.describe().loc['mean','max_diff']
    max_diff_std = df_m.describe().loc['std','max_diff']

    std_cutoffs = std_mean + (stddev_cutoff * std_std)
    max_diff_cutoffs = max_diff_mean + (stddev_cutoff * max_diff_std)

    print(f'Cutoffs for standard deviation = {round(std_cutoffs,4)}, and for maximum difference from mean = {round(max_diff_cutoffs,4)}')

    df = df_m[df_m['std'] < std_cutoffs] # 
    df = df[df['max_diff'] < max_diff_cutoffs]

    print(f'{len(df)} minutes of data remain after filtering:')
    
    ## Plotting:
    # Create a figure with a customized grid layout
    fig = plt.figure(figsize=(15, 6))
    gs = plt.GridSpec(1, 4, figure=fig, width_ratios=[3,1,1,1])
    
    # Create the first subplot for the lineplot (taking up half the width)
    ax1 = fig.add_subplot(gs[0,0])
    
    # Plot the three series as a line plot on the same axis
    ax1.plot(df['mean'], label='Mean V', color = 'blue')
    ax1.plot(df['std'], label='Std. Dev.', color = 'orange')
    ax1.plot(df['max_diff'], label='Max Diff.', color = 'green')

    # Set titles and labels
    ax1.set_title('Time Series of Magnetometer Data, Outliers Removed')
    ax1.set_xticks([])
    ax1.set_ylabel('Magnetometer Reading (V)')
    # Inside the same plotting code, replace the rotation line with:

    ax1.legend()

    # Create the next three subplots for the boxplots (each taking up 1/6 of the width)
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
    ax4 = fig.add_subplot(gs[0, 3], sharey=ax1)
    
    #
    avg_mean = str(round(df['mean'].mean(),4))
    avg_std = str(round(df['std'].mean(),4))
    avg_max_diff = str(round(df['max_diff'].mean(),4))

    # Plot the boxplots
    sns.boxplot(data=df['mean'], ax=ax2, color = 'blue')
    ax2.set_title('Mean V Distribution')
    ax2.set_ylabel('')
    ax2.set_xticks([0], labels = [f'Average Val = {avg_mean}'])

    sns.boxplot(data=df['std'], ax=ax3, color = 'orange')
    ax3.set_title('Std. Dev. Distribtion')
    ax3.set_ylabel('')
    ax3.set_xticks([0], labels = [f'Average Val = {avg_std}'])

    sns.boxplot(data=df['max_diff'], ax=ax4, color = 'green')
    ax4.set_title('Max Difference Distribution')
    ax4.set_ylabel('')
    ax4.set_xticks([0], labels = [f'Average Val = {avg_max_diff}'])

    # Adjust the layout
    plt.tight_layout()
    plt.show()     
    
    good_indices = set(df.index)
    all_indices = set(df_m.index)

    # Find the indices that are in df1 but not in df2
    df_m_cut_indices = list(all_indices - good_indices)
    
    return df.index, df_m_cut_indices
# --------------------------------------------------------------------------------------------------


# FUNCTIONS - ANALYSIS -----------------------------------------------------------------------------

def generate_psd_from_ts(time_series_data, sample_rate = 2597):
    '''
    Given time series data and a sample rate, this function transforms the data using a Fast Fourier Transform
    and returns an array with the Power Spectrum Density.
    '''
    fft_data = np.fft.fft(time_series_data)
    n = len(time_series_data)
    psd = (1 / (sample_rate * n)) * np.abs(fft_data[:n//2])**2
    freqs = np.fft.fftfreq(n, 1/sample_rate)[:n//2]

    return freqs, np.sqrt(2*psd)

def get_calibration_data(filename):
    '''
    Loads calibration data based on the calibration file type (.dat vs .csv)
    '''
    if filename[-3:] == 'dat':
        calibration_data = np.loadtxt(filename)
    elif filename[-3:] == 'csv':
        calibration_data = np.loadtxt(filename, delimiter = ",")
    else:
        print('Something is wrong, check your calibration file')
        return
    
    frequency_calibration = calibration_data[:, 0]  # Frequency values in Hz
    voltage_calibration = 10**-3 * calibration_data[:, 1]    # Volts per nano-Tesla (original calibration data is in mV/nT)
    phase_calibration = calibration_data[:, 2]
    return frequency_calibration, voltage_calibration, phase_calibration

def calibrate_frequency_domain(freq_domain, freqs, voltage_calibration, phase_correction):
    '''
    Calibrates the frequency domain transformation of the time domain data using magnetometer data.
    '''
    calibrated_freq_domain = np.zeros_like(freq_domain, dtype=complex)
    for i, freq in enumerate(freqs):
        if freq >= 0:  # Only process positive frequencies
            calibration_factor = voltage_calibration[i]
            phase = phase_correction[i]

            # Apply calibration factor and phase correction
            # the 2* is for one sided freq>0
            calibrated_freq_domain[i] = 2 * freq_domain[i] / calibration_factor * np.exp(1j * phase)
    return calibrated_freq_domain

def calibrate_data(freqs, psd_agg, calibration_filename):
    '''
    Takes in the aggregated psd data and frequencies and applies the calibration data to it.
    '''
    frequency_calibration, voltage_calibration, phase_calibration = get_calibration_data(calibration_filename)

    voltage_calibration_interpolated = np.interp(np.abs(freqs), frequency_calibration, voltage_calibration)
    phase_correction_interpolated = np.interp(np.abs(freqs), frequency_calibration, phase_calibration)

    calibrated_freq_domain = calibrate_frequency_domain(psd_agg, freqs, voltage_calibration_interpolated, phase_correction_interpolated)
    
    # Convert all to real here
    calibrated_freq_domain = [x.real for x in calibrated_freq_domain]
    
    return calibrated_freq_domain


# FUNCTIONS - GRAPHING -----------------------------------------------------------------------------
def graph_psd(psd_agg): #, start = 0, stop = -1):
    '''
    Note: Uses '1' as a placeholder for stopping at len(psd_agg)
    '''
    # if stop == -1:
    #     stop = len(psd_agg)
    # # Since Series index is the frequency, can simply plot the series
    plt.plot(psd_agg)#.iloc[start:stop])
    # s1 = round(start/60 + 1/60,4)
    # s2 = round(stop/60 + 1/60,4)
    plt.title('PSD')#(f'PSD from {s1} to {s2}')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()



# ARCHIVED FUNCTIONS ---------------------------------------------------------------------------------
def agg_psd(directory, sample_rate, df_m_clean_indices, calibration_data_file):
    '''
    Returns the aggregated power spectrum density.
    '''
    # Get the frequency bins using the sample rate
    freqs = np.fft.fftfreq(60*sample_rate, 1/sample_rate)[:(60*sample_rate)//2] 
    
    frequency_calibration, voltage_calibration, phase_calibration = get_calibration_data(calibration_data_file)
    
    voltage_calibration_interpolated = np.interp(np.abs(freqs), frequency_calibration, voltage_calibration) #v2

    # Initialize data holder as an array of zero values
    psd_avg = np.zeros(30*sample_rate)
    fft_avg = np.zeros(60*sample_rate, dtype = 'complex128') #v2

    # To ensure proper ordering of data
    file_list = [f for f in os.listdir(directory) if f.endswith(".h5")]
    file_list.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)))

    # For tracking progress
    counter = 0

    # Iterate over each file in the directory
    for filename in file_list:
        if filename.endswith(".h5"):
            file_path = os.path.join(directory, filename)
            x, timestamps, data = pull_record(file_path)

            # Check to see if it is in the filter
            if timestamps[0] in df_m_clean_indices:   

                data = flatten_data(data)

                n = len(data)

                # Compute the Fast Fourier Transform (FFT)
                fft_result = np.fft.fft(data)

                # Calculate the one-sided power spectral density
                psd = (1 / (sample_rate * n)) * np.abs(fft_result[:n//2])**2
                
                psd = np.sqrt(2*psd) #v2
                
                # Now Calibrate Here!
                #calibrated_psd = calibrate_data(freqs, psd, calibration_data_file)#v1
                calibrated_psd = psd / voltage_calibration_interpolated #v2

                # This line of code aggregates
                psd_avg += calibrated_psd

            # Update counter for progress tracking
                counter += 1
                print(f'Completed {counter} files', end = '\r')
                
    return pd.Series(psd_avg, index = freqs)

def get_interpolated_seconds_data(timestamps, data):
    '''
    For each second's worth of samples (2597) get the key descriptive data as a downsample.
    
    Returns the mean, std and max difference from mean for that second - the latter useful
    for outlier identification.
    '''
    # Create the data holder for the output (the average voltage for each second, indexed to a timestamp)
    avg_voltage = np.zeros(60) 
    
    # Create the array of rounded timestamps (to the second)
    timestamps_floor = [x.replace(microsecond=0) for x in timestamps]   
    
    # Now convert to seconds
    timestamps_floor_int = np.array([dt.timestamp() for dt in timestamps_floor])
    # and do the same for the original timestamps
    timestamps = np.array([dt.timestamp() for dt in timestamps])
        
    for i, second in enumerate(data):
        avg_voltage[i] = second.mean()
        
    # Now interpolate!
    avg_voltage_interp = np.interp(timestamps_floor_int, timestamps, avg_voltage)
        
    return timestamps_floor, avg_voltage_interp


def get_NON_interpolated_seconds_data(timestamps, data):
    '''
    For each second's worth of samples (2597) get the key descriptive data as a downsample.
    
    Returns the mean, std and max difference from mean for that second - the latter useful
    for outlier identification.
    '''
    # Create the data holder for the output (the average voltage for each second, indexed to a timestamp)
    avg_voltage = np.zeros(60) 
        
    for i, second in enumerate(data):
        avg_voltage[i] = second.mean()
    
    
    return avg_voltage