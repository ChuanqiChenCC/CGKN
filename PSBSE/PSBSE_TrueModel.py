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
################# Data Generation #################
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


def NRMSE(u, u_pred):
    return torch.mean(torch.sqrt(torch.mean((u-u_pred)**2, dim=0) / torch.var(u, dim=0))).item()


###################################################################################
################# Dynamics Prediction by Deterministic Part of SDE ################
###################################################################################

test_u_dot_pred = torch.zeros_like(test_u_dot)
test_u_dot_pred[:,0] = beta_x * test_u[:,0] + alpha * test_u[:,0] * test_u[:,1] + alpha * test_u[:,1] * test_u[:,2]
test_u_dot_pred[:,1] = beta_y * test_u[:,1] - alpha * test_u[:,0] ** 2 + 2 * alpha * test_u[:,0] * test_u[:,2]
test_u_dot_pred[:,2] = beta_z * test_u[:,2] - 3 * alpha * test_u[:,0] * test_u[:,1]
NRMSE(test_u_dot, test_u_dot_pred)


############################################################################
################# State Prediction by Ensemble Simulation  #################
############################################################################
torch.manual_seed(0)
dt_ens = 0.001
Nsim = 100
test_short_steps = int(0.1/dt)
test_u0 = test_u[::test_short_steps]
test_u_shortPred_ens = torch.zeros(Nsim, test_u0.shape[0], test_short_steps*10, test_u0.shape[1]) # (Nsim, N, t, x)
test_u_shortPred_ens[:, :, 0] = test_u0.repeat([Nsim, 1, 1])
for n in range(test_short_steps*10-1):
    test_u_shortPred_ens[:, :, n + 1, 0] = test_u_shortPred_ens[:, :, n, 0] + (beta_x * test_u_shortPred_ens[:,:,n,0] + alpha * test_u_shortPred_ens[:,:,n,0] * test_u_shortPred_ens[:,:,n,1] + alpha * test_u_shortPred_ens[:,:,n,1] * test_u_shortPred_ens[:,:,n,2]) * dt_ens + sigma_x * np.sqrt(dt_ens) * torch.randn(Nsim, test_u0.shape[0])
    test_u_shortPred_ens[:, :, n + 1, 1] = test_u_shortPred_ens[:, :, n, 1] + (beta_y * test_u_shortPred_ens[:,:,n,1] - alpha * test_u_shortPred_ens[:,:,n,0] ** 2 + 2 * alpha * test_u_shortPred_ens[:,:,n,0] * test_u_shortPred_ens[:,:,n,2]) * dt_ens + sigma_y * np.sqrt(dt_ens) * torch.randn(Nsim, test_u0.shape[0])
    test_u_shortPred_ens[:, :, n + 1, 2] = test_u_shortPred_ens[:, :, n, 2] + (beta_z * test_u_shortPred_ens[:,:,n,2] - 3 * alpha * test_u_shortPred_ens[:,:,n,0] * test_u_shortPred_ens[:,:,n,1]) * dt_ens + sigma_z * np.sqrt(dt_ens) * torch.randn(Nsim, test_u0.shape[0])
test_u_shortPred = test_u_shortPred_ens[:, :, ::10].mean(0).reshape(-1, 3)
NRMSE(test_u, test_u_shortPred)


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
SIG1 = torch.tensor([[sigma_x**2]])
SIG2 = torch.diag(torch.tensor([sigma_y**2, sigma_z**2]))
sig1 = torch.tensor([[sigma_x]])
sig2 = torch.diag(torch.tensor([sigma_y, sigma_z]))

J = 100
p = 2
test_u2_ens = torch.zeros((J, Ntest, p))
for n in range(Ntest-1):
    f1 = beta_y * test_u2_ens[:, n, 0] - alpha * test_u1[n] ** 2 + 2 * alpha * test_u1[n] * test_u2_ens[:, n, 1]
    f2 = beta_z * test_u2_ens[:, n, 1] - 3 * alpha * test_u1[n] * test_u2_ens[:, n, 0]
    f = torch.stack([f1, f2]).T
    g1 = beta_x * test_u1[n] + alpha * test_u1[n] * test_u2_ens[:, n, 0] + alpha * test_u2_ens[:, n, 0] * test_u2_ens[:, n, 1]
    g = torch.stack([g1]).T
    g_bar = torch.mean(g, dim=0)
    CCOV = cross_cov(test_u2_ens[:, n], g)
    test_u2_ens[:, n+1, :] = test_u2_ens[:,n,:] + f*dt + torch.randn(J,p) @ sig2 * np.sqrt(dt) - 0.5 * ((g+g_bar) * dt - 2 * (test_u1[n+1] - test_u1[n])) @ (CCOV @ torch.inverse(SIG1)).T

test_mu = torch.mean(test_u2_ens, dim=0)
test_R = torch.zeros((Ntest, 2, 2))
for n in range(Ntest):
    test_R[n] = torch.cov(test_u2_ens[:, n].T)
test_mu_std = torch.sqrt(torch.diagonal(test_R, dim1=1, dim2=2))

NRMSE(test_u[300:, 1:], test_mu[300:])

# np.save(path_abs + r"/Data/PSBSE_TrueModel_test_mu.npy", test_mu)
# np.save(path_abs + r"/Data/PSBSE_TrueModel_test_mu_std.npy", test_mu_std)

