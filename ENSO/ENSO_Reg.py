# Train Data: PDE Data1; Test Data: PDE Data2
# Observable Variable: TW,TC,TE, WB(external); Unobservable: HW, HC, HE
# Forecast Horizon: 12-Months; DA Horizon: 40 Years with 4 Years be cut-off
# Target Loss (3): AE Loss + Forecast Loss + DA Loss
# Forecast Loss: TW, TC, TE, HW, HC, HE with Path-wise MSE in Original Space
# Knowledge-based model for Normalized Data
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
train_ENSO_mat = sp.io.loadmat(r"Data/ENSO_model_data1.mat")
test_ENSO_mat = sp.io.loadmat(r"Data/ENSO_model_data2.mat")
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


############################################
########## System Identification ###########
############################################
idx_forecast = [0,1,2,4,5,6]

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

train_LibCG = torch.cat([train_u, train_u[:, :4]**2,
                         train_u[:, [0]]*train_u[:, 1:],
                         train_u[:, [1]]*train_u[:, 2:],
                         train_u[:, [2]]*train_u[:, 3:],
                         train_u[:, [3]]*train_u[:, 4:]], dim=-1)

CEM = cem(train_LibCG.numpy(), train_u_dot.numpy())
CEM.round(3)
CEI = CEM > np.concatenate([ [0.005]*3, [0.001],  [0.1]*3 ]).reshape(-1, 1)




###########################################
########## Parameter Estimation ###########
###########################################
CEI = np.c_[np.array([True]*7), CEI]
train_LibCG = torch.cat( [torch.ones(Ntrain, 1), train_LibCG], dim=-1 )
param_matrix = np.zeros( (CEI.shape[0], CEI.shape[1]) )
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
        LibCG = torch.cat([torch.ones(x.shape[0], 1),
                           x, x[:, :4] ** 2,
                           x[:, [0]] * x[:, 1:],
                           x[:, [1]] * x[:, 2:],
                           x[:, [2]] * x[:, 3:],
                           x[:, [3]] * x[:, 4:]], dim=-1)
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
NRMSE(test_u_dot[:, idx_forecast], test_u_dot_pred[:, idx_forecast])


# State Prediction
test_short_steps = int(12/12/dt)
test_u0 = test_u[::test_short_steps]
with torch.no_grad():
    test_u_shortPred = torchdiffeq.odeint(regmodel, test_u0, test_t[:test_short_steps])
test_u_shortPred = test_u_shortPred.permute(1,0,2).reshape(Ntest, -1)
nnF.mse_loss(test_u[:, idx_forecast], test_u_shortPred[:, idx_forecast])
NRMSE(test_u[:, idx_forecast], test_u_shortPred[:, idx_forecast])



# Lead Prediction
test_lead_steps = int(12/12/dt)
si = int(1000/dt)
ei = int(1100/dt)
test_u0 = test_u[si:ei-test_lead_steps]
with torch.no_grad():
    test_u_leadPred = torchdiffeq.odeint(regmodel, test_u0, test_t[:test_lead_steps])[-1]
nrmse = NRMSE(test_u[si+test_lead_steps:ei, idx_forecast], test_u_leadPred[:, idx_forecast])
corr = torch.corrcoef( torch.stack([test_u[si+test_lead_steps:ei, 2], test_u_leadPred[:, 2]]) )[0,1].item()



# Data Assimilation
aa = param_matrix[0, CEI[0]]
bb = param_matrix[1, CEI[1]]
cc = param_matrix[2, CEI[2]]
dd = param_matrix[3, CEI[3]]
ee = param_matrix[4, CEI[4]]
ff = param_matrix[5, CEI[5]]
gg = param_matrix[6, CEI[6]]
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
for n in range(Ntest-1):
    f1 = torch.tensor([ [aa[0] + aa[1]*test_u1[n,2] + aa[2]*test_u1[n,3]],
                        [bb[0] + bb[1]*test_u1[n,3]],
                        [cc[0]+cc[1]*test_u1[n,1]+cc[2]*test_u1[n,2]],
                        [dd[0]+dd[1]*test_u1[n,3]]])
    g1 = torch.tensor([ [0, aa[3], aa[4]],
                        [0, bb[2], bb[3]],
                        [0, 0, cc[3]],
                        [0, 0, 0]])
    f2 = torch.tensor([[ee[0] + ee[1]*test_u1[n,3]],
                       [ff[0]+ff[1]*test_u1[n,2]+ff[2]*test_u1[n,3]],
                       [gg[0]]])
    g2 = torch.tensor([ [ee[2], ee[3], ee[4]],
                        [0, 0, ff[3]],
                        [0, gg[1], 0]])
    du1 = test_u1[n+1] - test_u1[n]
    mu1 = mu0 + (f2+g2@mu0)*dt + (R0@g1.T) @ invs1os1 @ (du1 -(f1+g1@mu0)*dt)
    R1 = R0 + (g2@R0 + R0@g2.T + s2os2 - R0@g1.T@invs1os1@g1@R0)*dt
    test_mu_pred[n+1] = mu1
    test_R_pred[n+1] = R1
    mu0 = mu1
    R0 = R1

test_mu_pred = test_mu_pred.squeeze(-1)

NRMSE(test_u[4*360:, indices_u2], test_mu_pred[4*360:])

