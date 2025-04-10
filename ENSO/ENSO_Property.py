import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torchdiffeq
import time

device = "cpu"
torch.manual_seed(0)
np.random.seed(0)

plt.rc("text", usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

#########################################
############# Data Import ###############
#########################################

train_ENSO_mat = sp.io.loadmat("Data/ENSO_model_data1.mat")
test_ENSO_mat = sp.io.loadmat("Data/ENSO_model_data2.mat")
dt = 1/360

train_u = np.vstack([np.array(train_ENSO_mat[key]) for key in ["T_W_model", "T_C_model", "T_E_model", "wind_burst_model", "H_W_model", "H_C_model", "H_E_model" ]] ).T
train_u_dot = np.diff(train_u, axis=0)/dt
train_u_dot = np.vstack([train_u_dot, train_u_dot[-1]])
Ntrain = len(train_u)
train_t = np.arange(0, Ntrain*dt, dt)
train_u = torch.tensor(train_u, dtype=torch.float32)
train_u_dot = torch.tensor(train_u_dot, dtype=torch.float32)
train_t = torch.tensor(train_t, dtype=torch.float32)

test_u = np.vstack([np.array(test_ENSO_mat[key]) for key in ["T_W_model", "T_C_model", "T_E_model", "wind_burst_model", "H_W_model", "H_C_model", "H_E_model" ]] ).T
test_u_dot = np.diff(test_u, axis=0)/dt
test_u_dot = np.vstack([test_u_dot, test_u_dot[-1]])
Ntest = len(test_u)
test_t = np.arange(0, Ntest*dt, dt)
test_u = torch.tensor(test_u, dtype=torch.float32)
test_u_dot = torch.tensor(test_u_dot, dtype=torch.float32)
test_t = torch.tensor(test_t, dtype=torch.float32)

indices_u1 = np.arange(4)
indices_u2 = np.arange(4, 7)
dim_u1 = len(indices_u1)
dim_u2 = len(indices_u2)
idx_forecast = [0, 1, 2, 4, 5, 6]

class Normalizer:
    def __init__(self, x, eps=1e-5):
        # x is in the shape tensor (N, x)
        self.mean = torch.mean(x, dim=0)
        self.std = torch.std(x, dim=0)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x*(self.std+self.eps) + self.mean

normalizer = Normalizer(train_u)
train_u_normalized = normalizer.encode(train_u)
train_u_dot_normalized = torch.diff(train_u_normalized, dim=0)/dt
train_u_dot_normalized = torch.vstack([train_u_dot_normalized, train_u_dot_normalized[-1]])


#########################################################
############### Visualization for ENSO ##################
#########################################################
test_u = test_u.numpy()
test_t = test_t.numpy()

def acf(x, lags):
    i = np.arange(0, lags+1)
    v = np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  for i in range(1, lags+1)])
    return (i, v)


si = int(1000/dt)
ei = int(1100/dt)


fig = plt.figure(figsize=(10, 5))
# TE
ax00 = plt.subplot2grid((3, 5), (0, 0), colspan=3)
ax00.plot(test_t[si:ei], test_u[si:ei, 2], linewidth=1, color="black")
ax00.set_ylabel(r"$T_E$", fontsize=15, rotation=0)
ax00.set_title(r"\textbf{(a) True signal}", fontsize=15)
ax00.set_xlim([test_t[si], test_t[ei]])
ax00.set_ylim([-3, 5])
ax00.set_yticks([-2, 0, 2, 4])
ax01 = plt.subplot2grid((3, 5), (0, 3))
sns.kdeplot(test_u[:, 2], ax=ax01, linewidth=1, bw_adjust=2, color="black")
ax01.set_ylabel("")
ax01.set_xlim([-3, 5])
ax01.set_xticks([-2, 0, 2, 4])
ax01.set_yticks([0, 0.25, 0.5])
ax01.set_title(r"\textbf{(b) PDF}", fontsize=15)
ax02 = plt.subplot2grid((3, 5), (0, 4))
ax02.plot(acf(test_u[:, 2], int(5/dt))[0]*dt, acf(test_u[:, 2], int(5/dt))[1], linewidth=1, color="black")
ax02.set_xlim([0, 5])
ax02.set_xticks(np.arange(0, 6))
ax02.set_ylim([-0.6, 1.0])
ax02.set_yticks([-0.5, 0, 0.5, 1.0])
ax02.set_title(r"\textbf{(c) ACF}", fontsize=15)
# HW
ax10 = plt.subplot2grid((3, 5), (1, 0), colspan=3)
ax10.plot(test_t[si:ei], test_u[si:ei, 4], linewidth=1, color="black")
ax10.set_ylabel(r"$H_W$", fontsize=15, rotation=0)
ax10.set_xlim([test_t[si], test_t[ei]])
ax10.set_ylim([-60, 50])
ax10.set_yticks([-40, 0, 40])
ax11 = plt.subplot2grid((3, 5), (1, 3))
sns.kdeplot(test_u[:, 4], ax=ax11, linewidth=1, bw_adjust=2, color="black")
ax11.set_ylabel("")
ax11.set_yticks( [0, 0.02, 0.04] )
ax11.set_xlim([-60, 50])
ax11.set_xticks([-40, 0, 40])
ax12 = plt.subplot2grid((3, 5), (1, 4))
ax12.plot(acf(test_u[:, 4], int(5/dt))[0]*dt, acf(test_u[:, 4], int(5/dt))[1], linewidth=1, color="black")
ax12.set_xlim([0, 5])
ax12.set_xticks(np.arange(0, 6))
# ax12.set_ylim([-0.2, 1.0])
ax12.set_yticks([-0.5, 0, 0.5, 1.0])
# WB
ax20 = plt.subplot2grid((3, 5), (2, 0), colspan=3)
ax20.plot(test_t[si:ei], test_u[si:ei, 3], linewidth=1, color="black")
ax20.set_ylabel(r"$a_p$", fontsize=15, rotation=0)
ax20.set_xlim([test_t[si], test_t[ei]])
ax20.set_ylim([-25, 30])
ax20.set_yticks([-20, 0, 20])
ax20.set_xlabel(r"$t$", fontsize=15)
ax21 = plt.subplot2grid((3, 5), (2, 3))
sns.kdeplot(test_u[:, 3], ax=ax21, linewidth=1, bw_adjust=2, color="black")
ax21.set_ylabel("")
ax21.set_xlim([-25, 30])
ax21.set_xticks([-20, 0, 20])
ax21.set_yticks([0, 0.1, 0.2])
ax22 = plt.subplot2grid((3, 5), (2, 4))
ax22.plot(acf(test_u[:, 3], int(5/dt))[0]*dt, acf(test_u[:, 3], int(5/dt))[1], linewidth=1, color="black")
ax22.set_xlim([0, 5])
ax22.set_xticks(np.arange(0, 6))
ax22.set_ylim([-0.1, 1.0])
ax22.set_yticks([0, 0.5, 1.0])
ax22.set_xlabel(r"$t$", fontsize=15)
for ax in fig.get_axes():
    ax.tick_params(labelsize=14, length=5, width=0.5, direction="in", top=True, bottom=True, left=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1)
fig.tight_layout()
fig.savefig("Figs/ENSO_Property.png")
fig.savefig("Figs/ENSO_Property.pdf")

