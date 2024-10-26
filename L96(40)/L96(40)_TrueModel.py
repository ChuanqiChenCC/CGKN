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
dim_u1 = len(indices_u1)
dim_u2 = len(indices_u2)


def NRMSE(u, u_pred):
    return torch.mean(torch.sqrt(torch.mean((u-u_pred)**2, dim=0) / torch.var(u, dim=0))).item()
#####################################################################################
################# Dynamics Prediction by Deterministic Part of SDE  #################
#####################################################################################

test_u_dot_pred = torch.zeros_like(test_u_dot)
for i in range(I):
    test_u_dot_pred[:, i] = -test_u[:, i] + test_u[:,(i+1)%I]*test_u[:,i-1] - test_u[:,i-2]*test_u[:,i-1] + F

NRMSE(test_u_dot, test_u_dot_pred)

############################################################################
################# State Prediction by Ensemble Simulation  #################
############################################################################
torch.manual_seed(0)
dt_ens = 0.001
Nsim = 100
test_short_steps = int(0.2/dt)
test_u0 = test_u[::test_short_steps]
test_u_shortPred_ens = torch.zeros(Nsim, test_u0.shape[0], test_short_steps*10, test_u0.shape[1]) # (Nsim, N, t, x)
test_u_shortPred_ens[:, :, 0] = test_u0.repeat([Nsim, 1, 1])
for n in range(test_short_steps*10-1):
    for i in range(I):
        u_dot = -test_u_shortPred_ens[:, :, n, i] + test_u_shortPred_ens[:, :, n, (i+1)%I]*test_u_shortPred_ens[:,:, n,i-1] - test_u_shortPred_ens[:,:,n,i-2]*test_u_shortPred_ens[:,:,n,i-1] + F
        test_u_shortPred_ens[:,:,n+1,i] = test_u_shortPred_ens[:,:,n,i] + u_dot*dt_ens + sigma*np.sqrt(dt_ens)*torch.randn(test_u_shortPred_ens.shape[0], test_u_shortPred_ens.shape[1])

test_u_shortPred = test_u_shortPred_ens[:, :, ::10].mean(0).reshape(-1, I)
nnF.mse_loss(test_u, test_u_shortPred)
NRMSE(test_u, test_u_shortPred)

################################################################################
################# Lead-Time Prediction by Ensemble Simulation  #################
################################################################################
np.random.seed(0)
si = int(100/dt)
ei = int(110/dt)
test_lead_steps = int(0.2/dt)
Nsim = 100
test_u0 = test_u[si:ei-test_lead_steps]
test_u_leadPred_ens = np.zeros((Nsim, test_u0.shape[0], test_lead_steps, test_u0.shape[1])) # (Nsim, N, t, x)
test_u_leadPred_ens[:, :, 0] = np.repeat(test_u0[np.newaxis, :, :], Nsim, axis=0)
for n in range(test_lead_steps-1):
    for i in range(I):
        u_dot = -test_u_leadPred_ens[:, :, n, i] + test_u_leadPred_ens[:, :, n, (i+1)%I]*test_u_leadPred_ens[:,:, n,i-1] - test_u_leadPred_ens[:,:,n,i-2]*test_u_leadPred_ens[:,:,n,i-1] + F
        test_u_leadPred_ens[:,:,n+1,i] = test_u_leadPred_ens[:,:,n,i] + u_dot*dt + sigma*np.sqrt(dt)*np.random.randn(test_u_leadPred_ens.shape[0], test_u_leadPred_ens.shape[1])

test_u_leadPred = test_u_leadPred_ens[:,:, -1].mean(0)

np.mean( (test_u[si+test_lead_steps:ei, 1] - test_u_leadPred[:, 1])**2 )
np.corrcoef( np.stack([test_u[si+test_lead_steps:ei, 1], test_u_leadPred[:, 1]]) )

# np.save(path_abs + r"/Data/L96(40)_TrueModel_test_u_leadPred0u2(400uTo410u).npy", test_u_leadPred)


####################################################################################
################# Data Assimilation by Ensemble Kalman-Bucy Filter #################
####################################################################################

torch.manual_seed(0)
def cross_cov(X, Y):
    n = X.shape[0]
    assert n == Y.shape[0]
    X_centered = X - torch.mean(X, dim=0)
    Y_centered = Y - torch.mean(Y, dim=0)
    cross_cov_matrix = X_centered.T @ Y_centered / (n - 1)
    return cross_cov_matrix

test_u1 = test_u[:, indices_u1]
SIG1 = torch.diag(torch.tensor([sigma**2]*dim_u1))
SIG2 = torch.diag(torch.tensor([sigma**2]*dim_u2))
sig1 = torch.diag(torch.tensor([sigma]*dim_u1))
sig2 = torch.diag(torch.tensor([sigma]*dim_u2))

J = 100
p = dim_u2
test_u2_ens = torch.zeros((J, Ntest, p))
for n in range(Ntest-1):
    print("time step:", n+1)
    test_u1_repeat = test_u1[n].repeat((J, 1))
    test_u_ens = torch.stack([test_u1_repeat, test_u2_ens[:, n]], dim=-1).reshape(-1, I)
    test_u_dot_ens = torch.zeros_like(test_u_ens)
    for i in range(I):
        test_u_dot_ens[:, i] = -test_u_ens[:, i] + test_u_ens[:, (i + 1) % I] * test_u_ens[:, i - 1] - test_u_ens[:, i - 2] * test_u_ens[:, i - 1] + F
    f = test_u_dot_ens[:, 1::2]
    g = test_u_dot_ens[:, ::2]
    g_bar = torch.mean(g, dim=0)
    CCOV = cross_cov(test_u2_ens[:, n], g)
    Sys_term = f*dt + torch.randn(J,p)@sig2 * np.sqrt(dt)
    DA_term = -0.5*((g+g_bar)*dt-2*(test_u1[n+1]-test_u1[n])) @ (CCOV@torch.inverse(SIG1)).T
    test_u2_ens[:, n+1, :] = test_u2_ens[:, n, :] + Sys_term + DA_term

test_mu = torch.mean(test_u2_ens, dim=0)
test_R = torch.zeros((test_u2_ens.shape[1], test_u2_ens.shape[2], test_u2_ens.shape[2]))
for n in range(test_u2_ens.shape[1]):
    test_R[n] = torch.cov(test_u2_ens[:, n, :].T)
test_mu_std = torch.sqrt(torch.diagonal(test_R, dim1=1, dim2=2))

NRMSE(test_u[200:, indices_u2], test_mu[200:]) # Cut-Point is the same as CGKN


# np.save(path_abs + r"/Data/L96(40)_TrueModel_test_mu.npy", test_mu)
# np.save(path_abs + r"/Data/L96(40)_TrueModel_test_mu_std.npy", test_mu_std)
