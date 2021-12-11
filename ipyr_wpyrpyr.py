from cell import Cell
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from scipy.signal import find_peaks, welch, butter, filtfilt
from tqdm import tqdm
import os
# =====================================================================
# Functions
def simulate(time, C):
    
    #euler_integrate
    for t in range(len(time)-1):
        C.r_pyr[t+1]=C.r_pyr[t] + dt*(C.alpha_pyr)*(-C.r_pyr[t] + r_o*f(C.wpyrpyr*C.r_pyr[t-tau] + C.wbicpyr*C.r_bic[t-tau] + C.wpvpyr*C.r_pv[t-tau] + C.i_pyr)) + np.sqrt(2*C.alpha_pyr*C.D_pyr*dt)*np.random.normal(0,1)
        C.r_bic[t+1]=C.r_bic[t]+dt*(C.alpha_bic)*(-C.r_bic[t]+r_o*f(C.wpyrbic*C.r_pyr[t-tau]+C.i_bic))+np.sqrt(2*C.alpha_bic*C.D_bic*dt)*np.random.normal(0,1)
        C.r_cck[t+1]=C.r_cck[t]+dt*(C.alpha_cck)*(-C.r_cck[t]+r_o*f(C.wcckcck*C.r_cck[t-tau]+C.wpvcck*C.r_pv[t-tau]+C.i_cck))+np.sqrt(2*C.alpha_cck*C.D_cck*dt)*np.random.normal(0,1)
        C.r_pv[t+1]=C.r_pv[t]+dt*(C.alpha_pv)*(-C.r_pv[t]+r_o*f(C.wcckpv*C.r_cck[t-tau]+C.wpvpv*C.r_pv[t-tau]+C.wpyrpv*C.r_pyr[t-tau]+C.i_pv))+np.sqrt(2*C.alpha_pv*C.D_pv*dt)*np.random.normal(0,1)
    return C


def bandPassFilter(data, cutoff, fs, order=5):
    # nyquist frequency
    nyq = 0.5 * fs
    band = cutoff / nyq
    b, a = butter(order, band, btype = 'band', analog = False)
    y = filtfilt(b, a, data)
    return y

def calc_spectral(cell, dt, band = 'theta', mode = 'peak_freq', plot_Fig = False, plot_Filter = False):
    # choose which band of oscillations you want to filter
    if band == 'theta':
        cutoff = np.array([3, 15])
    elif band == 'gamma':
        # should this be changed - maybe?
        cutoff = np.array([15, 100])
        
    r_pyr_filt = bandPassFilter(cell.r_pyr, cutoff, 1/dt)
    r_bic_filt = bandPassFilter(cell.r_bic, cutoff, 1/dt)
    r_cck_filt = bandPassFilter(cell.r_cck, cutoff, 1/dt)
    r_pv_filt = bandPassFilter(cell.r_pv, cutoff, 1/dt)
    
    if plot_Filter:
        plt.figure()
        plt.plot(time[plot_start_time:], r_pyr_filt[plot_start_time:], label = 'PYR')
        plt.plot(time[plot_start_time:], r_bic_filt[plot_start_time:], label = 'BIC')
        plt.plot(time[plot_start_time:], r_cck_filt[plot_start_time:], label = 'CCK')
        plt.plot(time[plot_start_time:], r_pv_filt[plot_start_time:], label = 'PV')
        plt.legend()
    
    # create periodograms of the filtered cell traces
    freq_pyr, welch_pyr = welch(r_pyr_filt, fs = 1/dt, nperseg='1024')
    freq_bic, welch_bic = welch(r_bic_filt, fs = 1/dt, nperseg='1024')
    freq_cck, welch_cck = welch(r_cck_filt, fs = 1/dt, nperseg='1024')
    freq_pv, welch_pv = welch(r_pv_filt, fs = 1/dt, nperseg='1024')
    
    if plot_Fig: 
        plt.figure()
        plt.plot(freq_pyr, welch_pyr, label = 'PYR')
        plt.plot(freq_bic, welch_bic, label = 'BIC')
        plt.plot(freq_cck, welch_cck, label = 'CCK')
        plt.plot(freq_pv, welch_pv, label = 'PV')
        plt.xlim(0, 100)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Spectral Power')
        plt.legend()
    
    # find the peaks from the periodograms - correspond to peak frequency and power
    peaks_pyr, props_pyr = find_peaks(welch_pyr)
    peaks_bic, props_bic = find_peaks(welch_bic)
    peaks_cck, props_cck = find_peaks(welch_cck)
    peaks_pv, props_pv = find_peaks(welch_pv)
    
#     print(np.argmax(welch_pyr[peaks_pyr]))
    if mode == 'peak_freq':
        fm_pyr = freq_pyr[peaks_pyr][np.argmax(welch_pyr[peaks_pyr])]
        fm_bic = freq_bic[peaks_bic][np.argmax(welch_bic[peaks_bic])]
        fm_cck = freq_cck[peaks_cck][np.argmax(welch_cck[peaks_cck])]
        fm_pv = freq_pv[peaks_pv][np.argmax(welch_pv[peaks_pv])]
        
        return np.array([fm_pyr, fm_bic, fm_cck, fm_pv])
    elif mode == 'power':
        power_pyr = np.max(welch_pyr[peaks_pyr])
        power_bic = np.max(welch_bic[peaks_bic])
        power_cck = np.max(welch_cck[peaks_cck])
        power_pv = np.max(welch_pv[peaks_pv])
        
        return np.array([power_pyr, power_bic, power_cck, power_pv])
#     plt.axvline(fm_pyr)
    
    
def normalise_heatmap(hmap, cutoff = 0.0):
    norm_map = np.zeros_like(hmap)
    for i in range(len(hmap)):
        max_val = np.max(hmap[i])
        norm_map[i] = hmap[i] / max_val
    
    return norm_map

def plot_trace(time, cell, plot_start_time):
    plt.plot(time[plot_start_time:], cell.r_pyr[plot_start_time:], label = 'PYR')
    plt.plot(time[plot_start_time:], cell.r_bic[plot_start_time:], label = 'BIC')
    plt.plot(time[plot_start_time:], cell.r_cck[plot_start_time:], label = 'CCK')
    plt.plot(time[plot_start_time:], cell.r_pv[plot_start_time:], label = 'PV')

    plt.xlabel('Time (ms)')
    plt.ylabel('Activity')
    plt.legend()
# =====================================================================
# Parameters
T=2.0 # total time (units in sec)
dt=0.001 # plotting and Euler timestep (parameters adjusted accordingly)

# FI curve
beta=10
tau=5
h=0
r_o=30
# ======================================================================
new_cell = Cell()
f = lambda u: 1/(1+np.exp(-beta*(u-h)))

# create time array
time=np.arange(0,T,dt)
plot_start_time = 3 * time.size // 4
# initialise instance of Cell for simulation
new_cell._set_init_state(len(time))
new_cell = simulate(time, new_cell)
# ======================================================================
grid_size = 20
res = str(int(grid_size**2 // 100))
px = 'ipyr'; py = 'wpyrpyr'

i_pyr = np.linspace(0, 0.5, grid_size)
wpyrpyr = np.linspace(0, 0.05, grid_size)
p_space = np.meshgrid(i_pyr, wpyrpyr)

# base_values = [0.07, 0.03]
# new_cell._set_connections()
# new_cell.i_pyr, new_cell.wpyrpyr = base_values
# new_cell._set_init_state(len(time))
# new_cell = simulate(time, new_cell)

# base_freq = np.zeros((4, 2)); base_power = np.zeros((4, 2))
# base_power[:, 0] = calc_spectral(new_cell, dt, mode = 'power', plot_Fig = False)
# base_power[:, 1] = calc_spectral(new_cell, dt, band = 'gamma', mode = 'power', plot_Fig = False)
# base_freq[:, 0] = calc_spectral(new_cell, dt, mode = 'peak_freq', plot_Fig = False)
# base_freq[:, 1] = calc_spectral(new_cell, dt, band = 'gamma', mode = 'peak_freq', plot_Fig = False)

# create arrays to store power and frequency values for each point in parameter space
theta_power = np.zeros((4, grid_size, grid_size))
gamma_power = np.zeros((4, grid_size, grid_size))
theta_freq = np.zeros((4, grid_size, grid_size))
gamma_freq = np.zeros((4, grid_size, grid_size))

for i in tqdm(range(len(wpyrpyr))):
    for j in range(len(i_pyr)):
        new_cell._set_connections() # reset connections
        # set parameters
        new_cell.i_pyr = i_pyr[j] 
        new_cell.wpyrpyr = wpyrpyr[i]
        new_cell._set_init_state(len(time)) # initialise cell state for sim
        new_cell = simulate(time, new_cell) 
        # store values
        theta_power[:, i, j] = calc_spectral(new_cell, dt, mode = 'power', plot_Fig = False)
        gamma_power[:, i, j] = calc_spectral(new_cell, dt, band = 'gamma', mode = 'power', plot_Fig = False)
        theta_freq[:, i, j] = calc_spectral(new_cell, dt, mode = 'peak_freq', plot_Fig = False)
        gamma_freq[:, i, j] = calc_spectral(new_cell, dt, band = 'gamma', mode = 'peak_freq', plot_Fig = False)

# normalise power
norm_theta_power = normalise_heatmap(theta_power)
norm_gamma_power = normalise_heatmap(gamma_power)

# Create Plots/Heatmaps
label = ['PYR', 'BiC', 'CCK', 'PV']

dir = f'./Figures/{px}_{py}/org2'

try:
    os.mkdir(dir)
except FileExistsError:
    pass

x_label = '$i_{pyr}$'
y_label = '$w_{pyr, pyr}$'

# Theta Power
for i in range(4):    
    fig, ax = plt.subplots()
    plt.grid(False)
    c = ax.pcolormesh(p_space[0], p_space[1], theta_power[i], cmap = 'viridis')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('{} Theta Power'.format(label[i]))
    fig.colorbar(c)
    plt.savefig(f'{dir}/{res}_{label[i]}_{px}_{py}_theta_power.png')

# Gamma Power
for i in range(4):    
    fig, ax = plt.subplots()
    plt.grid(False)
    c = ax.pcolormesh(p_space[0], p_space[1], gamma_power[i], cmap = 'viridis')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('{} Gamma Power'.format(label[i]))
    fig.colorbar(c)
    plt.savefig(f'{dir}/{res}_{label[i]}_{px}_{py}_gamma_power.png')

# Normalised Power Difference
for i in range(4):    
    fig, ax = plt.subplots()
    plt.grid(False)
    c = ax.pcolormesh(p_space[0], p_space[1], norm_theta_power[i] - norm_gamma_power[i], cmap='Spectral')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('{} Normalised Power Difference'.format(label[i]))
    fig.colorbar(c)
    plt.savefig(f'{dir}/{res}_{label[i]}_{px}_{py}_norm_power_diff.png')

# Theta Frequency
for i in range(4):    
    fig, ax = plt.subplots()
    plt.grid(False)
    c = ax.pcolormesh(p_space[0], p_space[1], theta_freq[i], cmap = 'viridis')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('{} Theta Freq'.format(label[i]))
    fig.colorbar(c)
    plt.savefig(f'{dir}/{res}_{label[i]}_{px}_{py}_theta_freq.png')

# Theta Frequency with Power Contours
for i in range(4):    
    fig, ax = plt.subplots()
    plt.grid(False)
    c = ax.pcolormesh(p_space[0], p_space[1], theta_freq[i], cmap = 'viridis')
    cs= ax.contour(p_space[0], p_space[1], theta_power[i], cmap = "inferno", alpha = 1, levels = 6)
    ax.clabel(cs, inline=True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('{} Theta Freq'.format(label[i]))
    fig.colorbar(c)
    plt.savefig(f'{dir}/{res}_{label[i]}_{px}_{py}_theta_freq_contour.png')

# Gamma Frequency
for i in range(4):    
    fig, ax = plt.subplots()
    plt.grid(False)
    c = ax.pcolormesh(p_space[0], p_space[1], gamma_freq[i], cmap = 'viridis')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('{} Gamma Freq'.format(label[i]))
    fig.colorbar(c)
    plt.savefig(f'{dir}/{res}_{label[i]}_{px}_{py}_gamma_freq.png')

# Gamma Frequency with Power Contours
for i in range(4):    
    fig, ax = plt.subplots()
    plt.grid(False)
    c = ax.pcolormesh(p_space[0], p_space[1], gamma_freq[i], cmap = 'viridis')
    cs= ax.contour(p_space[0], p_space[1], gamma_power[i], cmap = "inferno", alpha = 1, levels = 5)
    ax.clabel(cs, inline=True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('{} Gamma Freq'.format(label[i]))
    fig.colorbar(c)
    plt.savefig(f'{dir}/{res}_{label[i]}_{px}_{py}_gamma_freq_contour.png')
