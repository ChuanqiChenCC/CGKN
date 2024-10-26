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


############################################
########## System Identification ###########
############################################

def cem(A, B):
    """
    :param A: numpy.array(Nt, Na); Basis Functions
    :param B: numpy.array(Nt, Nb); Dynamics
    :return: numpy.array(Nb, Na); Causation Entropy Matrix C(a->b|[B\b])
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

train_LibCG = torch.stack([train_u[:, 0], train_u[:, 1], train_u[:, 2],train_u[:, 0]**2, train_u[:, 0]*train_u[:, 1], train_u[:, 0]*train_u[:, 2]], dim=-1)

CEM = cem(train_LibCG.numpy(), train_u_dot.numpy())
CEM.round(3)

CEI = CEM.round(3) > 0.1

###########################################
########## Parameter Estimation ###########
###########################################

param_matrix = np.zeros_like(CEM)
for i in range(len(param_matrix)):
    X = train_LibCG[:, CEI[i]].numpy()
    y = train_u_dot[:, i].numpy()
    param_matrix[i, CEI[i]] = np.linalg.inv(X.T@X)@X.T@y
param_matrix = torch.tensor(param_matrix, dtype=torch.float32)

train_u_dot_pred = train_LibCG @ param_matrix.T
sigma_hat = torch.sqrt( dt*torch.mean( (train_u_dot - train_u_dot_pred)**2, dim=0 ) )

class RegModel(nn.Module):
    def __init__(self, param_matrix):
        super().__init__()
        self.param_matrix = param_matrix
    def forward(self, t, x):
        LibCG = torch.stack([x[:, 0], x[:, 1], x[:, 2],
                             x[:, 0]**2, x[:, 0]*x[:, 1], x[:, 0]*x[:, 2]]).T
        x_dot = LibCG @ self.param_matrix.T
        return x_dot

###################################################
########## Test Causal-based Regression ###########
###################################################
def NRMSE(u, u_pred):
    return torch.mean(torch.sqrt(torch.mean((u-u_pred)**2, dim=0) / torch.var(u, dim=0))).item()

regmodel = RegModel(param_matrix)

# Dynamics Prediction
with torch.no_grad():
    test_u_dot_pred = regmodel(None, test_u)

NRMSE(test_u_dot, test_u_dot_pred)


# State Prediction
test_short_steps = int(0.1/dt)
test_u0 = test_u[::test_short_steps]
with torch.no_grad():
    test_u_shortPred = torchdiffeq.odeint(regmodel, test_u0, test_t[:test_short_steps])
test_u_shortPred = test_u_shortPred.permute(1, 0, 2).reshape(Ntest, -1)
NRMSE(test_u, test_u_shortPred)



# Data Assimilation
a = param_matrix[0, CEI[0]]
b = param_matrix[1, CEI[1]]
c = param_matrix[2, CEI[2]]
dim_u1 = len(indices_u1)
dim_u2 = len(indices_u2)
test_u1 = test_u[:, indices_u1].unsqueeze(-1)
s1, s2 = torch.diag(sigma_hat[indices_u1]), torch.diag(sigma_hat[indices_u2])
invs1os1 = torch.linalg.inv(s1@s1.T)
s2os2 = s2@s2.T
test_mu_pred = torch.zeros((Ntest, dim_u2, 1))
test_R_pred = torch.zeros((Ntest, dim_u2, dim_u2))
mu0 = torch.zeros((dim_u2, 1))
R0 = 0.01*torch.eye(dim_u2)
test_mu_pred[0] = mu0
test_R_pred[0] = R0
for n in range(1, Ntest):
    f1 = torch.zeros(1, 1)
    g1 = torch.tensor([[a*test_u1[n-1, 0], 0]])
    f2 = torch.tensor([ [ b[0]*test_u1[n-1, 0]**2], [0] ])
    g2 = torch.tensor([ [0, b[1]*test_u1[n-1, 0]], [ c*test_u1[n-1, 0], 0 ] ])
    du1 = test_u1[n] - test_u1[n-1]
    mu1 = mu0 + (f2+g2@mu0)*dt + (R0@g1.T) @ invs1os1 @ (du1 -(f1+g1@mu0)*dt)
    R1 = R0 + (g2@R0 + R0@g2.T + s2os2 - R0@g1.T@invs1os1@g1@R0)*dt
    test_mu_pred[n] = mu1
    test_R_pred[n] = R1
    mu0 = mu1
    R0 = R1
test_mu_pred = test_mu_pred.squeeze(-1)
NRMSE(test_u[300:, indices_u2], test_mu_pred[300:])

test_mu_std_pred = torch.diagonal(test_R_pred, dim1=1, dim2=2)

# np.save(path_abs + r"/Data/PSBSE_Reg_test_mu_pred.npy", test_mu_pred)
# np.save(path_abs + r"/Data/PSBSE_Reg_test_mu_std_pred.npy", test_mu_std_pred)
