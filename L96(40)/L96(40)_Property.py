import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torchdiffeq
import time

path_abs = r"C:\Users\chenc\CodeProject\CGKN\L96(40)"
device = "cpu"
torch.manual_seed(0)
np.random.seed(0)

plt.rc("text", usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


######################################
########## Data Generation ###########
######################################
F = 8
sigma = 0.5

I = 40
Lt = 500
dt = 0.001
Nt = int(Lt/dt) + 1
t = np.linspace(0, Lt, Nt)
u = np.zeros((Nt, I))
u[0] = np.random.randn(I)
# for n in range(Nt-1):
#     for i in range(I):
#         u_dot = -u[n, i] + u[n,(i+1)%I]*u[n,i-1] - u[n,i-2]*u[n,i-1] + F
#         u[n+1, i] = u[n, i] + u_dot*dt + sigma*np.sqrt(dt)*np.random.randn()

u = np.load(path_abs + "/Data/L96(40)_SimData.npy")

# Sub-sampling
u = u[::10]
dt = 0.01
Nt = int(Lt/dt) + 1
t = np.linspace(0, Lt, Nt)
u_dot = np.diff(u, axis=0)/dt

# Split data in to train and test
u_dot = torch.tensor(u_dot, dtype=torch.float32)
u = torch.tensor(u[:-1], dtype=torch.float32)
t = torch.tensor(t[:-1], dtype=torch.float32)

Ntrain = int(u.shape[0] * 0.6)
Ntest = int(u.shape[0] * 0.4)
train_u = u[:Ntrain]
train_u_dot = u_dot[:Ntrain]
train_t = t[:Ntrain]
test_u = u[-Ntest:]
test_u_dot = u_dot[-Ntest:]
test_t = t[-Ntest:]

indices_u1 = np.arange(0, I, 2)
indices_u2 = np.arange(1, I, 2)

#########################################
########## Data Visualization ###########
#########################################

u = train_u.numpy()
t = train_t.numpy()

u_part = u[10000:15000]
t_part = t[10000:15000]

def acf(x, lag):
    i = np.arange(0, lag+1)
    v = np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  for i in range(1, lag+1)])
    return (i, v)

np.sum( acf(u[:, 0], 100)[1] ) *dt
np.sum( acf(u[:, 1], 100)[1] ) *dt
np.sum( acf(u[:, 2], 100)[1] ) *dt
np.sum( acf(u[:, 3], 100)[1] ) *dt



fig = plt.figure(figsize=(6, 6))
# Pcolor
ax00 = plt.subplot2grid((3, 4), (0, 0), colspan=4)
c = ax00.pcolor(t_part[::10], np.arange(1, I+1), u_part[::10].T, cmap="jet")
cbar = fig.colorbar(c, ax=ax00, location='top')
cbar.ax.tick_params(labelsize=12, length=3, width=1)
ax00.set_title(r"\textbf{(a) Hovmoller diagram of true simulation}", fontsize=13, pad=45)
ax00.set_xlabel(r"$t$", fontsize=13, labelpad=-8)
ax00.set_ylabel(r"$i$", fontsize=13, rotation=0)
ax00.set_yticks(np.array([1, 20, 40]))
ax00.set_xticks(np.arange(100, 150+10, 10))
# Time Series
ax10 = plt.subplot2grid((3, 4), (1, 0), colspan=4)
ax10.plot(t_part, u_part[:, 1], linewidth=1, color="black")
ax10.set_title(r"\textbf{(b) True signal}", fontsize=13)
ax10.set_xlabel(r"$t$", fontsize=13, labelpad=-5)
ax10.set_ylabel(r"$x_2$", fontsize=13, rotation=0)
ax10.set_xlim([100, 150])
ax10.set_ylim([np.min(u), np.max(u)])
ax10.set_yticks([-6, 0, 6, 12])
# PDF & ACF
ax20 = plt.subplot2grid((3, 4), (2, 0), colspan=2)
sns.kdeplot(u[:, 1], ax=ax20, linewidth=1, bw_adjust=2, color="black")
ax20.set_ylabel("")
ax20.set_yticks([0, 0.05, 0.1])
ax20.set_xlim([np.min(u), np.max(u)])
ax20.set_xticks([-6, 0, 6, 12])
ax20.set_title(r"\textbf{(c) PDF}", fontsize=13)
ax21 = plt.subplot2grid((3, 4), (2, 2), colspan=2)
ax21.plot(np.linspace(0, 1, 100+1),  acf(u[:, 1], 100)[1], linewidth=1, color="black")
ax21.set_xlabel(r"$t$", fontsize=13)
ax21.set_yticks([0, 0.5, 1])
ax21.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax21.set_title(r"\textbf{(d) ACF}", fontsize=13)
ax21.set_xlim([0, 1])
ax21.set_ylim([-0.2, 1.0])
for ax in fig.get_axes():
    ax.tick_params(labelsize=12, length=5, width=0.8, direction="in", top=True, bottom=True, left=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1)
for spine in ax00.spines.values():
    spine.set_linewidth(0)
ax00.tick_params(labelsize=12, length=5, width=0.8, direction="out", top=False, bottom=True, left=True, right=False)
cbar.ax.tick_params(labelsize=12, length=5, width=0.8, direction="out", top=True, bottom=False, left=False, right=False)
fig.tight_layout()
fig.subplots_adjust(wspace=2.5, hspace=0.5)
fig.savefig(path_abs + "/Figs/L96(40)_Property.pdf")
fig.savefig(path_abs + "/Figs/L96(40)_Property.png")


