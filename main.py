%matplotlib ipympl
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from resutils import graph2json as g2json
from mpl_toolkits.mplot3d import Axes3D
from resutils import percolator
from resutils import netfitter2 as netfitter
from resutils import graph2json as g2json
from resutils import utilities
from resutils import plott
import networkx as nx
from tqdm import tqdm_notebook, tnrange
import time
import pandas as pd
from collections import OrderedDict
import json
import scipy
from scipy.signal import chirp, spectrogram
import copy
import requests
import gzip
import base64
import time
import igraph as ig
import random
from scipy import signal
# from tqdm import tqdm

perc = percolator.Percolator(serverUrl=" http://landau-nic0.mse.ncsu.edu:15142/percolator/")
nf_cpu = netfitter.NetworkFitter(serverUrl="http://landau-nic0.mse.ncsu.edu:15122/symphony/")
nf_lancuda = netfitter.NetworkFitter(serverUrl="http://landau-nic0.mse.ncsu.edu:15122/symphony/")

import pandas as pd
df = pd.read_csv('/home/jovyan/work/LabMeasurements/Damien/Ag_c_pos.txt')

lab_voltage = df['Drive(V)']
lab_voltage = list(lab_voltage)

lab_current = df['Current(mA)']
lab_current = list(lab_current)
lab_current = lab_current[1:]

lab_time = df['Time(s)']
lab_time = list(lab_time)
sim_time = lab_time[-1]

def run_sim(y,circ,sim_time):
    y = y
    circ = circ
    sim_time = sim_time
    dic={}
    for inputid in circ['inputids']:
        dic[inputid]=list(y)

    int_time = 1e-4 #1e-3 > int_time
    time_val = (sim_time/1640)
    circ=g2json.modify_integration_time(circ, set_val=str(int_time)) #'0.05'
    utils = utilities.Utilities(serverUrl=nf_lancuda.serverUrl)
    key=nf_lancuda.init_steps(circ['circuit'],utils)
    utils.settings(key,peekInterval=time_val,pokeInterval=time_val) #0.5e-6 1000 samples
    utils.setArbWaveData(key,dic)
    utils.setMeasurableElements(key,circ['outputids'])
    # utils.startForAndWait(key,T)

    utils.startForAndWait(key,sim_time)
    meas=json.loads(utils.measurements_gzip(key))

    nf_lancuda.complete_steps(key,utils)
#     fig,m_table=plotly_meas(meas,title='asdfasdf',value='current',ivt_scale=1)
    t,m_table = get_t_mtable(meas)
    return m_table


import plotly.graph_objects as go


def plot_meas(meas, value='current'):
    t = []
    m_table = {}
    if type(meas) == list:
        new_meas = {}
        new_meas['measurements'] = meas
        meas = new_meas

    for step in meas['measurements']:
        t.append(step['time'])
        for record_key in step['records'].keys():
            if record_key not in m_table.keys():
                m_table[record_key] = []
            m_table[record_key].append(step['records'][record_key][value])

    plt.figure()

    for measurable in m_table.keys():
        plt.plot(t, m_table[measurable], label=str(measurable))

    plt.xlabel('Time (s)')
    plt.ylabel('Current (A)')
    plt.legend()
    plt.show()
    return m_table


def get_t_mtable(meas, value='current'):
    t = []
    m_table = {}
    if type(meas) == list:
        new_meas = {}
        new_meas['measurements'] = meas
        meas = new_meas

    for step in meas['measurements']:
        t.append(step['time'])
        for record_key in step['records'].keys():
            if record_key not in m_table.keys():
                m_table[record_key] = []
            m_table[record_key].append(step['records'][record_key][value])

    return t, m_table


def plotly_meas(meas, value='current', title='', ivt_scale=1):
    assert ivt_scale > 0., "ivt_scale should be > 0"

    t, m_table = get_t_mtable(meas, value)
    #     plt.figure()
    fig = go.Figure()
    for measurable in m_table.keys():
        fig.add_trace(
            go.Scatter(x=(np.array(t) * ivt_scale).tolist(), y=(np.array(m_table[measurable]) / ivt_scale).tolist(),
                       mode='lines',
                       name=str(measurable)
                       ))
    #         plt.plot(t,m_table[measurable],label=str(measurable))
    fig.update_layout(title=title,
                      xaxis_title='Time (s)',
                      yaxis_title='Current (A)')

    #     plt.xlabel('Time (s)')
    #     plt.ylabel('Current (A)')
    #     plt.legend()
    #     plt.show()
    return fig, m_table

circ={'circuit': '{"0": ["$", 1, "5e-05", 27.727228452313398, 72, 1.0, 50], "1": ["195", 0, 1, 0, 1, "1.0E9", "1.0E11", "3.0E-10", "400.0E-9", "100.0E-9", "30", "1.0E9", "0.5", "2.0", "-1.0", "1.0E9"], "2": ["r", 1, 2, 0, 2, "100.0"], "3": ["g", 2, 3, 0, 0], "4": ["R", 0, 4, 0, 3, "0", "40.0", "5.0", "0.0", "0.0", "0.5"]}',
      'inputids': [3],
      'outputids': [2],
      'controlids': []}

circ['circuit'] = '{{"0": ["$", 1, "5e-05", 27.727228452313398, 72, 1.0, 50], "1": ["195", 0, 1, 0, 1, "1.0E9", "1.0E11", "3.0E-10", "{}", "{}", "{}", "1.0E9", "0.5", "{}", "-1.0", "1.0E9"], "2": ["r", 1, 2, 0, 2, "100.0"], "3": ["g", 2, 3, 0, 0], "4": ["R", 0, 4, 0, 3, "0", "40.0", "5.0", "0.0", "0.0", "0.5"]}}'.format(4e-9,2e-12,1,4)
m_table = run_sim(lab_voltage,circ,sim_time)
sim_current = m_table[str(circ['outputids'][0])]
sim_current = sim_current[1:]
# zip_object = zip(lab_current, sim_current)
# for list1_i, list2_i in zip_object:
#     current_diff.append((list1_i-list2_i)**2)
current_diff_score=np.linalg.norm(np.array(sim_current)-np.array(lab_current))

import multiprocessing as mp
import tqdm
circ = g2json.modify_integration_time(circ, set_val='1e-4')
# mp.pool.ThreadPool(processes=)
with mp.pool.ThreadPool(processes=100) as pool:
    outvals7 = pool.starmap(map_single_mnist_ravel,zip(list(bin_xtrain[:]),y_train[:]))

import optuna


def objective(trial):
    # total_widths = [16000e-9 ] #nm
    total_widths = [1000e-9, 2000e-9, 3000e-9, 3500e-9, 4000e-9, 4200e-9, 4500e-9, 5000e-9, 6000e-9, 8000e-9, 10000e-9,
                    12000e-9, 14000e-9, 16000e-9]  # nm
    mobilities = [1e-12, 1.5e-12, 1.75e-12, 2e-12, 2.5e-12, 3e-12, 4e-12, 4.5e-12, 5e-12, 6e-12, 7e-12, 8e-12, 9e-12,
                  10e-12, 12e-12, 16e-12, 32e-12, 64e-12, 128e-12, 256e-12, 512e-12, 800e-12, 900e-12, 1000e-12,
                  1100e-12, 1200e-12, 2000e-12, 3000e-12, 5000e-12, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6]  # um^2
    noise_factors = [0.001, 0.002, 0.009, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 4, 5, 10, 20, 21, 22, 23, 24, 25, 26,
                     27, 28, 29, 30, 50, 100, 200, 400, 600, 1000, 2000, 4000, 8000, 10000]
    R_on_max = [2, 5]
    tao = [10, 20, 60, 120, 700, 1e9]
    rho_list = [1e9]  # slope during on state

    tw = trial.suggest_categorical("tw", total_widths)
    mob = trial.suggest_categorical("mob", mobilities)
    nf = trial.suggest_categorical("nf", noise_factors)
    tao_val = trial.suggest_categorical("tao_val", tao)
    R_max = trial.suggest_categorical("R_max", R_on_max)
    rho = trial.suggest_categorical("rho", rho_list)
    cur_diff_scores = []

    def calc_atom(a):
        circ[
            'circuit'] = '{{"0": ["$", 1, "5e-05", 27.727228452313398, 72, 1.0, 50], "1": ["195", 0, 1, 0, 1, "1.0E9", "1.0E11", "3.0E-10", "{}", "{}", "{}", "{}", "0.5", "{}", "-1.0", "{}"], "2": ["r", 1, 2, 0, 2, "100.0"], "3": ["g", 2, 3, 0, 0], "4": ["R", 0, 4, 0, 3, "0", "40.0", "5.0", "0.0", "0.0", "0.5"]}}'.format(
            tw, mob, nf, tao_val, R_max, rho)
        m_table = run_sim(lab_voltage, circ, sim_time)
        sim_current = m_table[str(circ['outputids'][0])]
        sim_current = sim_current[1:]
        current_diff_score = np.linalg.norm(np.array(sim_current) - np.array(lab_current))
        return current_diff_score

    with mp.pool.ThreadPool(processes=10) as pool:
        cur_diff_scores = pool.starmap(calc_atom, zip(range(10)))
    return np.mean(cur_diff_scores)


study = optuna.create_study()
study.optimize(objective, n_trials=10000, n_jobs=5)

study.best_params  # E.g. {'x': 2.002108042}

#circ['circuit'] = '{{"0": ["$", 1, "5e-05", 27.727228452313398, 72, 1.0, 50], "1": ["195", 0, 1, 0, 1, "1.0E9", "1.0E11", "3.999E-11", "{}", "{}", "{}", "{}", "0.5", "{}", "-1.0", "{}"], "2": ["r", 1, 2, 0, 2, "100.0"], "3": ["g", 2, 3, 0, 0], "4": ["R", 0, 4, 0, 3, "0", "40.0", "5.0", "0.0", "0.0", "0.5"]}}'.format(4.2e-6,1.2e-9,24,0.25e9,5,2e9)
circ['circuit'] = '{{"0": ["$", 1, "5e-05", 27.727228452313398, 72, 1.0, 50], "1": ["195", 0, 1, 0, 1, "1.0E9", "1.0E11", "3.999E-11", "{}", "{}", "{}", "{}", "0.5", "{}", "-1.0", "{}"], "2": ["r", 1, 2, 0, 2, "100.0"], "3": ["g", 2, 3, 0, 0], "4": ["R", 0, 4, 0, 3, "0", "40.0", "5.0", "0.0", "0.0", "0.5"]}}'.format(8e-6,5e-9,21,1000000000,5,1000000000)
m_table = run_sim(lab_voltage,circ,sim_time)
currents = m_table[str(circ['outputids'][0])]

plt.style.use('classic')

fig,ax = plt.subplots()
fig.patch.set_facecolor('w')
ax.margins(0.05)


plt.plot(lab_voltage[1:], currents[1:],'-',color="tab:blue")
#plt.plot(voltage,currents)
plt.ylabel("Current (A)")
plt.xlabel("Voltage (V)")
plt.show()
plt.tight_layout()


df = pd.DataFrame({'voltage':lab_voltage[1:],'currents':currents[1:],'time':lab_time[1:]})
df.to_csv('single_wire_optuna_trial_5.csv')