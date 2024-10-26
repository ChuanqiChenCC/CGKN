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

############################################
########## System Identifycation ###########
############################################

def cem(A, B):
    """
    :param A: numpy.array(Nt, Na); Basis Functions
    :param B: numpy.array(Nt, Nb); Dynamics
    :return: numpy.array(Nb, Na); Causation Entropy Matrix C(X->Y|Z)
    """
    Na = A.shape[1]
    Nb = B.shape[1]
    CEM = np.zeros((Nb, Na))
    for i in range(Nb):
        XYZ = np.concatenate([A, B[:, [i]]], axis=1)
        RXYZ = np.cov(XYZ.T)
        RXYZ_det = np.linalg.det(RXYZ)
        RXZ = RXYZ[:-1, :-1]
        RXZ_det = np.linalg.det(RXZ)
        for j in range(Na):
            RYZ = np.delete(np.delete(RXYZ, j, axis=0), j, axis=1)
            RYZ_det = np.linalg.det(RYZ)
            RZ = RYZ[:-1, :-1]
            RZ_det = np.linalg.det(RZ)
            CEM[i, j] = 1/2 * np.log(RYZ_det) - 1/2*np.log(RZ_det) - 1/2*np.log(RXYZ_det) + 1/2*np.log(RXZ_det)
    return CEM

def basisCG1(x):
    # x: shape(N, 5)
    out = torch.stack([x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4],
                    x[:, 0]**2, x[:, 2]**2, x[:, 4]**2,
                    x[:,0]*x[:,1], x[:,0]*x[:,2], x[:,0]*x[:,3], x[:,0]*x[:,4],
                    x[:,1]*x[:,2], x[:,1]*x[:,4],
                    x[:,2]*x[:,3], x[:,2]*x[:,4],
                    x[:,3]*x[:,4]]).T
    return out

def basisCG2(x):
    # x: shape(N, 5)
    out = torch.stack([x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4],
                    x[:, 1]**2, x[:, 3]**2,
                    x[:,0]*x[:,1], x[:,0]*x[:,3],
                    x[:,1]*x[:,2], x[:,1]*x[:,3], x[:,1]*x[:,4],
                    x[:,2]*x[:,3],
                    x[:,3]*x[:,4]]).T
    return out

train_LibCG1_cat = torch.cat( [basisCG1( train_u[:, [i-2, i-1, i, i+1, (i+2)%I]] ) for i in indices_u1])
train_u_dot1_cat = train_u_dot[:, indices_u1].T.reshape(-1, 1)
CEM1 = cem(train_LibCG1_cat.numpy(), train_u_dot1_cat.numpy())
CEM1.round(3)
CEI1 = CEM1 > 0.01


train_LibCG2_cat = torch.cat( [basisCG2( train_u[:, [i-2, i-1, i, (i+1)%I, (i+2)%I]] ) for i in indices_u2])
train_u_dot2_cat = train_u_dot[:, indices_u2].T.reshape(-1, 1)
CEM2 = cem(train_LibCG2_cat.numpy(), train_u_dot2_cat.numpy())
CEM2.round(3)
CEI2 = CEM2 > 0.01

###########################################
########## Parameter Estimation ###########
###########################################

# Parameters for u1
param_matrix1 = np.zeros( (CEM1.shape[0], CEM1.shape[1]+1) )
X = train_LibCG1_cat[:, CEI1.flatten()].numpy()
X = np.c_[np.ones((X.shape[0], 1)), X]
y = train_u_dot1_cat.numpy()
mask = np.concatenate([np.array([True]), CEI1.flatten()])
param_matrix1[0, mask] = (np.linalg.inv(X.T@X)@X.T@y).flatten()
param_matrix1 = torch.tensor(param_matrix1, dtype=torch.float32)


# Parameters for u2
param_matrix2 = np.zeros( (CEM2.shape[0], CEM2.shape[1]+1) )
X = train_LibCG2_cat[:, CEI2.flatten()].numpy()
X = np.c_[np.ones( (X.shape[0], 1) ), X]
y = train_u_dot2_cat.numpy()
mask = np.concatenate([np.array([True]), CEI2.flatten()])
param_matrix2[0, mask] = (np.linalg.inv(X.T@X)@X.T@y).flatten()
param_matrix2 = torch.tensor(param_matrix2, dtype=torch.float32)


# Sigma Estimationj
train_u_dot1_pred_cat = torch.cat([torch.ones(train_LibCG1_cat.shape[0], 1), train_LibCG1_cat], dim=1) @ param_matrix1.T
train_u_dot1_pred = train_u_dot1_pred_cat.flatten().reshape(-1, Ntrain).T
train_u_dot2_pred_cat = torch.cat([torch.ones(train_LibCG2_cat.shape[0], 1), train_LibCG2_cat], dim=1) @ param_matrix2.T
train_u_dot2_pred = train_u_dot2_pred_cat.flatten().reshape(-1, Ntrain).T

train_u_dot_pred = torch.stack( [train_u_dot1_pred, train_u_dot2_pred], dim=-1).reshape(Ntrain, I)
sigma_hat = torch.sqrt( dt*torch.mean( (train_u_dot - train_u_dot_pred)**2, dim=0 ) )

class RegModel(nn.Module):
    def __init__(self, basisCG1, basisCG2, param_matrix1, param_matrix2):
        super().__init__()
        self.basisCG1 = basisCG1
        self.basisCG2 = basisCG2
        self.param_matrix1 = param_matrix1
        self.param_matrix2 = param_matrix2

    def forward(self, t, x):
        batch_size = x.shape[0]
        I = 40
        indices_u1 = np.arange(0, I, 2)
        indices_u2 = np.arange(1, I, 2)
        LibCG1_cat = torch.cat( [self.basisCG1( x[:, [i-2, i-1, i, i+1, (i+2)%I]] ) for i in indices_u1])
        LibCG2_cat = torch.cat( [self.basisCG2( x[:, [i-2, i-1, i, (i+1)%I, (i+2)%I]] ) for i in indices_u2])
        x_dot1_cat = torch.cat([torch.ones(LibCG1_cat.shape[0], 1), LibCG1_cat], dim=1) @ self.param_matrix1.T
        x_dot1 = x_dot1_cat.flatten().reshape(-1, batch_size).T
        x_dot2_cat = torch.cat([torch.ones(LibCG2_cat.shape[0], 1), LibCG2_cat], dim=1) @ self.param_matrix2.T
        x_dot2 = x_dot2_cat.flatten().reshape(-1, batch_size).T
        x_dot = torch.stack( [x_dot1, x_dot2], dim=-1).reshape(batch_size, I)
        return x_dot

###################################################
########## Test Causal-based Regression ###########
###################################################
def NRMSE(u, u_pred):
    return torch.mean(torch.sqrt(torch.mean((u-u_pred)**2, dim=0) / torch.var(u, dim=0))).item()

regmodel = RegModel(basisCG1, basisCG2, param_matrix1, param_matrix2)

# Dynamics Prediction
with torch.no_grad():
    test_u_dot_pred = regmodel(None, test_u)
NRMSE(test_u_dot, test_u_dot_pred)


# State Prediction
test_short_steps = int(0.2/dt)
test_u0 = test_u[::test_short_steps]
with torch.no_grad():
    test_u_shortPred = torchdiffeq.odeint(regmodel, test_u0, test_t[:test_short_steps], method="rk4", options={"step_size":0.001})
test_u_shortPred = test_u_shortPred.permute(1, 0, 2).reshape(Ntest, -1)
nnF.mse_loss(test_u, test_u_shortPred)
NRMSE(test_u, test_u_shortPred)


# Lead-Time Prediction
si = int(100/dt)
ei = int(150/dt)
test_lead_steps = int(0.2/dt)
test_u0 = test_u[si:ei-test_lead_steps]
with torch.no_grad():
    test_u_leadPred = torchdiffeq.odeint(regmodel, test_u0, t[:test_lead_steps], method="rk4", options={"step_size":0.001})[-1]

# np.save(path_abs + r"/Data/L96(40)_Reg_test_u_leadPred0u2(400uTo450u).npy", test_u_leadPred)


# Data Assimilation
a = param_matrix1[0, np.concatenate([np.array([True]), CEI1.flatten()])]
b = param_matrix2[0, np.concatenate([np.array([True]), CEI2.flatten()])]
dim_u1 = len(indices_u1)
dim_u2 = len(indices_u2)
test_u1 = test_u[:, indices_u1].unsqueeze(-1)
s1, s2 = torch.diag(sigma_hat[indices_u1]), torch.diag(sigma_hat[indices_u2])
invs1os1 = torch.linalg.inv(s1@s1.T)
s2os2 = s2@s2.T
test_mu_pred = torch.zeros((Ntest, dim_u2, 1))
test_R_pred = torch.zeros((Ntest, dim_u2, dim_u2))
mu0 = torch.zeros( (dim_u2, 1) )
R0 = 0.01*torch.eye(dim_u2)
test_mu_pred[0] = mu0
test_R_pred[0] = R0
for n in range(1, Ntest):
    # f1, g1, f2, g2 = [e.squeeze(0) for e in cgkn.cgn(u1[n-1].T)]
    f1 = a[0] + a[1]*test_u1[n-1]
    g1 = torch.zeros(dim_u1, dim_u2)
    mask1 = torch.arange(dim_u1).reshape(-1, 1)
    mask2 = torch.stack([torch.arange(-1, dim_u1-1), torch.arange(dim_u1)]).T
    g1[mask1, mask2] = torch.cat([a[3]*test_u1[n-1][torch.arange(-1, dim_u1-1)] + a[4]*test_u1[n-1], a[2]+a[5]*test_u1[n-1]], dim=1)
    f2 = b[0] + b[3]*test_u1[n-1]*test_u1[n-1][torch.arange(1, dim_u2+1)%dim_u2]
    g2 = torch.zeros(dim_u2, dim_u2)
    mask1 = torch.arange(dim_u2).reshape(-1, 1)
    mask2 = torch.stack([torch.arange(-1, dim_u2-1), torch.arange(dim_u2)]).T
    g2[mask1, mask2] = torch.cat([b[2]*test_u1[n-1], b[1].repeat([dim_u2, 1])], dim=1)
    du1 = test_u1[n] - test_u1[n-1]
    mu1 = mu0 + (f2+g2@mu0)*dt + (R0@g1.T) @ invs1os1 @ (du1 -(f1+g1@mu0)*dt)
    R1 = R0 + (g2@R0 + R0@g2.T + s2os2 - R0@g1.T@invs1os1@g1@R0)*dt
    test_mu_pred[n] = mu1
    test_R_pred[n] = R1
    mu0 = mu1
    R0 = R1

test_mu_pred = test_mu_pred.squeeze(-1)
NRMSE(test_u[200:, indices_u2], test_mu_pred[200:]) # Same cut-point as CGKN

test_mu_std_pred = torch.sqrt(test_R_pred.diagonal(dim1=-2, dim2=-1))

# np.save(path_abs + r"/Data/L96(40)_Reg_test_mu_pred.npy", test_mu_pred)
# np.save(path_abs + r"/Data/L96(40)_Reg_test_mu_std_pred.npy", test_mu_std_pred)
