import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import seaborn as sns
from helpers import plot_bode_mimo, plot_phase_mimo
import errno, os

class Parameters:
    ...

class Data:
    ...

# change this to look at different runs
run = 'MIMO Model/run 2/run_latent_z_results_T50_W1000_M2000'


if Path(run + '_traces.pkl').is_file():
    traces = pickle.load(open(run + '_traces.pkl', 'rb'))
else: 
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), run + '_traces.pkl') # fail loudly
if Path(run + '_par.pkl').is_file():
    par = pickle.load(open(run + '_par.pkl', 'rb'))
else: 
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), run + '_par.pkl') # fail loudly
if Path(run + '_data.pkl').is_file():
    sim_data = pickle.load(open(run + '_data.pkl', 'rb'))
else: 
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), run + '_data.pkl') # fail loudly

# y_hat = np.transpose(traces['y_hat_out'],(2,1,0))
y_hat = traces['y_hat_out']
PT = y_hat.shape[2]
ts = np.arange(PT)
err1 = np.zeros((y_hat.shape[0],y_hat.shape[2]))
err2 = np.zeros((y_hat.shape[0],y_hat.shape[2]))
for ii in ts:
    err1[:,ii] = y_hat[:,0,ii] - sim_data.y[0,ii]
    err2[:,ii] = y_hat[:,1,ii] - sim_data.y[1,ii]

fig = plt.figure()
fig.set_size_inches(6, 3)
plt.subplot(2, 1, 1)
plt.plot(ts,err1.mean(axis = 0), color=u'#1f77b4',linewidth = 1,label='Mean of error')
plt.fill_between(ts,np.percentile(err1,97.5,axis=0),np.percentile(err1,2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% interval')
plt.plot(ts,sim_data.y[0,0:PT]-sim_data.i[0,0:PT], color=u'#1f0000',linestyle='--',linewidth = 0.8,label='Measured value')
plt.axhline(y=0, color='r', linestyle='-', linewidth=0.8, label='True value')
plt.xlim([0,len(ts)])
plt.xlabel('Output 1')

plt.subplot(2, 1, 2)
plt.plot(ts,err2.mean(axis = 0), color=u'#1f77b4',linewidth = 1,label='Mean of error')
plt.fill_between(ts,np.percentile(err2,97.5,axis=0),np.percentile(err2,2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% interval')
plt.plot(ts,sim_data.y[1,0:PT]-sim_data.i[1,0:PT], color=u'#1f0000',linestyle='--',linewidth = 0.8,label='Measured value')
plt.axhline(y=0, color='r', linestyle='-', linewidth=0.8,label= 'True value')
plt.xlim([0,len(ts)])
plt.suptitle('Prediction error 4th order Hammerstein-Wiener')
plt.xlabel('Output 2')
plt.tight_layout()
plt.show()

omega = np.logspace(-2,np.log10(np.pi),200)
# plot_bode(traces['A'],traces['B'][:,:,[0]],traces['C'][:,[0],:],traces['D'][:,[0],[0]],par.A,par.B[:,[0]],par.C[[0],:],par.D[[0],[0]],omega)

plot_bode_mimo(traces['A'],traces['B'],traces['C'],traces['D'],par.A,par.B,par.C,par.D,2,2,omega)
# plot_phase_mimo(traces['A'],traces['B'],traces['C'],traces['D'],par.A,par.B,par.C,par.D,2,2,omega)

f, axe = plt.subplots(ncols=4, nrows=2)
f.set_size_inches(6, 2.4)
ax = sns.kdeplot(traces['alpha'][:,1], shade=True, ax = axe[0,0])
ax.axvline(par.sat_lower1, color='r', lw=1, linestyle='--')
ax.set(ylabel=None)
ax.set(xlabel='Sat. min.')
ax.set(yticks=[])
ax.set(yticklabels=[])
ax = sns.kdeplot(traces['alpha'][:,2], shade=True, ax = axe[0,1])
ax.axvline(par.dzone_left1, color='r', lw=1, linestyle='--')
ax.set(ylabel=None)
ax.set(xlabel='Deadz. left')
ax.set(yticks=[])
ax.set(yticklabels=[])
ax = sns.kdeplot(traces['alpha'][:,3], shade=True, ax = axe[0,2])
ax.axvline(par.dzone_right1, color='r', lw=1, linestyle='--')
ax.set(ylabel=None)
ax.set(xlabel='Deadz. right')
ax.set(yticks=[])
ax.set(yticklabels=[])
ax = sns.kdeplot(traces['alpha'][:,0], shade=True, ax = axe[0,3])
ax.axvline(par.sat_upper1, color='r', lw=1, linestyle='--')
ax.set(ylabel=None)
ax.set(xlabel='Sat. max.')
ax.set(yticks=[])
ax.set(yticklabels=[])


# f2, axe = plt.subplots(ncols=2, nrows=2)
# f2.set_size_inches(6.4, 2)
ax = sns.kdeplot(traces['beta'][:,3], shade=True, ax = axe[1,0])
ax.axvline(par.sat_lower2, color='r', lw=1, linestyle='--')
ax.set(ylabel=None)
ax.set(xlabel='Sat. min.')
ax.set(yticks=[])
ax.set(yticklabels=[])
ax = sns.kdeplot(traces['beta'][:,0], shade=True, ax = axe[1,1])
ax.axvline(par.dzone_left2, color='r', lw=1, linestyle='--')
ax.set(ylabel=None)
ax.set(xlabel='Deadz. left')
ax.set(yticks=[])
ax.set(yticklabels=[])
ax = sns.kdeplot(traces['beta'][:,1], shade=True, ax = axe[1,2])
ax.axvline(par.dzone_right2, color='r', lw=1, linestyle='--')
ax.set(ylabel=None)
ax.set(xlabel='Deadz. right')
ax.set(yticks=[])
ax.set(yticklabels=[])
ax = sns.kdeplot(traces['beta'][:,2], shade=True, ax = axe[1,3])
ax.axvline(par.sat_upper2, color='r', lw=1, linestyle='--')
ax.set(ylabel=None)
ax.set(xlabel='Sat. max.')
ax.set(yticks=[])
ax.set(yticklabels=[])
# plt.suptitle(r'Kernel density estimates for $\beta$')
plt.suptitle(r'Kernel density estimates for $\alpha$ & $\beta$, 4th order system')
plt.tight_layout()
plt.show()


1+1
...