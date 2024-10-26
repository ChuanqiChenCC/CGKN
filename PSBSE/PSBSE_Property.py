import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torchdiffeq
import time

path_abs = r"C:\Users\chenc\CodeProject\CGKN\PSBSE"
device = "cpu"
torch.manual_seed(0)
np.random.seed(0)

plt.rc("text", usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

###################################################
################# Data Geneartion #################
###################################################
beta_x, beta_y, beta_z = 0.2, -0.3, -0.5
sigma_x, sigma_y, sigma_z = 0.3, 0.5, 0.5
alpha = 5

Lt = 1000
dt = 0.001
Nt = int(Lt/dt) + 1
t = np.linspace(0, Lt, Nt)
u = np.zeros((Nt, 3))
u[0] = np.ones(3)
# for n in range(Nt-1):
#     u[n + 1, 0] = u[n, 0] + (beta_x * u[n,0] + alpha * u[n,0] * u[n,1] + alpha * u[n,1] * u[n,2]) * dt + sigma_x * np.sqrt(dt) * np.random.randn()
#     u[n + 1, 1] = u[n, 1] + (beta_y * u[n,1] - alpha * u[n,0] ** 2 + 2 * alpha * u[n,0] * u[n,2]) * dt + sigma_y * np.sqrt(dt) * np.random.randn()
#     u[n + 1, 2] = u[n, 2] + (beta_z * u[n,2] - 3 * alpha * u[n,0] * u[n,1]) * dt + sigma_z * np.sqrt(dt) * np.random.randn()

u = np.load(path_abs + "/Data/PSBSE_SimData.npy")

# Sub-sampling
u = u[::10]
dt = 0.01
Nt = int(Lt/dt) + 1
t = np.linspace(0, Lt, Nt)
u_dot = np.diff(u, axis=0)/dt

# Split data into train and test
u_dot = torch.tensor(u_dot, dtype=torch.float32)
u = torch.tensor(u[:-1], dtype=torch.float32)
t = torch.tensor(t[:-1], dtype=torch.float32)
Ntrain = int(u.shape[0]*0.8)
Ntest = int(u.shape[0]*0.2)
train_u = u[:Ntrain]
train_u_dot = u_dot[:Ntrain]
train_t = t[:Ntrain]
test_u = u[-Ntest:]
test_u_dot = u_dot[-Ntest:]
test_t = t[-Ntest:]

indices_u1 = np.array([0])
indices_u2 = np.array([1, 2])

u = train_u.numpy()
t = train_t.numpy()
#########################################
########## Data Visualization ###########
#########################################
#########################################

def acf(x, lags):
    i = np.arange(0, lags+1)
    v = np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  for i in range(1, lags+1)])
    return (i, v)


si = int(300/dt)
ei = int(400/dt)
fig = plt.figure(figsize=(10, 5))
# x dynamic
ax00 = plt.subplot2grid((3, 5), (0, 0), colspan=3)
ax00.plot(t[si:ei], u[si:ei, 0], linewidth=1, color="black")
ax00.set_ylabel(r"$x$", fontsize=15, rotation=0, labelpad=5)
ax00.set_title(r"\textbf{(a) True signal}", fontsize=15)
ax00.set_xlim([t[si], t[ei]])
ax01 = plt.subplot2grid((3, 5), (0, 3))
sns.kdeplot(u[:, 0], ax=ax01, linewidth=1, bw_adjust=2, color="black")
ax01.set_ylabel("")
ax01.set_xlim( [np.min(u[si:ei,0]), np.max(u[si:ei,0])] )
ax01.set_ylim([0, 2.1])
ax01.set_yticks([0, 1, 2])
ax01.set_title(r"\textbf{(b) PDF}", fontsize=15)
ax02 = plt.subplot2grid((3, 5), (0, 4))
ax02.plot(acf(u[:, 0], 1500)[0]*dt, acf(u[:, 0], 1500)[1], linewidth=1, color="black")
ax02.set_xlim([0, 15])
ax02.set_xticks(np.arange(0, 15+3, 3))
ax02.set_ylim([-0.2, 1.0])
ax02.set_yticks([0, 0.5, 1.0])
ax02.set_title(r"\textbf{(c) ACF}", fontsize=15)
# ax02.hlines(y=0.6, xmin=0, xmax=t[205], colors='red', linewidth=1, linestyles='dashed')
# ax02.vlines(x=t[205], ymin=0, ymax=0.6, colors='red', linewidth=1, linestyles='dashed')
# y dynamics
ax10 = plt.subplot2grid((3, 5), (1, 0), colspan=3)
ax10.plot(t[si:ei], u[si:ei, 1], linewidth=1, color="black")
ax10.set_ylabel(r"$y$", fontsize=15, rotation=0, labelpad=3)
ax10.set_xlim([t[si], t[ei]])
ax10.set_ylim([-2.4, 1.4])
ax10.set_yticks([-2, -1, 0, 1])
ax11 = plt.subplot2grid((3, 5), (1, 3))
sns.kdeplot(u[:, 1], ax=ax11, linewidth=1, bw_adjust=2, color="black")
ax11.set_ylabel("")
ax11.set_yticks( [0, 0.25, 0.5, 0.75] )
ax11.set_xlim( [-2.4, 1.4] )
ax11.set_xticks([-2, -1, 0, 1])
ax12 = plt.subplot2grid((3, 5), (1, 4))
ax12.plot(acf(u[:, 1], 1500)[0]*dt, acf(u[:, 1], 1500)[1], linewidth=1, color="black")
ax12.set_xlim([0, 15])
ax12.set_xticks(np.arange(0, 15+3, 3))
ax12.set_ylim([-0.2, 1.0])
ax12.set_yticks([0, 0.5, 1.0])
# ax12.hlines(y=0.6, xmin=0, xmax=t[73], colors='red', linewidth=1, linestyles='dashed')
# ax12.vlines(x=t[73], ymin=0, ymax=0.6, colors='red', linewidth=1, linestyles='dashed')
# z dynamics
ax20 = plt.subplot2grid((3, 5), (2, 0), colspan=3)
ax20.plot(t[si:ei], u[si:ei, 2], linewidth=1, color="black")
ax20.set_ylabel(r"$z$", fontsize=15, rotation=0, labelpad=3)
ax20.set_xlim([t[si], t[ei]])
ax20.set_xlabel(r"$t$", fontsize=15)
ax21 = plt.subplot2grid((3, 5), (2, 3))
sns.kdeplot(u[:, 2], ax=ax21, linewidth=1, bw_adjust=2, color="black")
ax21.set_ylabel("")
ax21.set_xlim( [np.min(u[si:ei,2]), np.max(u[si:ei,2])] )
ax21.set_yticks([0, 0.5, 1.0])
ax22 = plt.subplot2grid((3, 5), (2, 4))
ax22.plot(acf(u[:, 2], 1500)[0]*dt, acf(u[:, 2], 1500)[1], linewidth=1, color="black")
ax22.set_xlim([0, 15])
ax22.set_xticks(np.arange(0, 15+3, 3))
ax22.set_ylim([-0.2, 1.0])
ax22.set_yticks([0, 0.5, 1.0])
ax22.set_xlabel(r"$t$", fontsize=15)
# ax22.hlines(y=0.6, xmin=0, xmax=t[10], colors='red', linewidth=1, linestyles='dashed')
# ax22.vlines(x=t[10], ymin=0, ymax=0.6, colors='red', linewidth=1, linestyles='dashed')
for ax in fig.get_axes():
    ax.tick_params(labelsize=14, length=5, width=0.8, direction="in", top=True, bottom=True, right=True, left=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1)
fig.tight_layout()
fig.savefig(path_abs + "/Figs/PSBSE_Property.pdf")
fig.savefig(path_abs + "/Figs/PSBSE_Property.png")
