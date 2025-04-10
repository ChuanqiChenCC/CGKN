# Train Data: PDE Data1; Test Data: PDE Data2
# Observable Variable: TW,TC,TE, WB(external); Unobservable: HW, HC, HE
# Forecast Horizon: 12-Months; DA Horizon: 40 Years with 4 Years be cut-off
# Target Loss (3): AE Loss + Forecast Loss + DA Loss
# Forecast Loss: TW, TC, TE, HW, HC, HE with Path-wise MSE in Original Space
# Knowledge-based regression model for Normalized Data

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
idx_forecast = [0,1,2,4,5,6]


############################################
########## System Identification ###########
############################################

def cem(A, B):
    Na = A.shape[1]
    Nb = B.shape[1]
    CEM = torch.zeros(Nb, Na)
    for i in range(Nb):
        XYZ = torch.cat([ A, B[:, [i]] ], dim=-1)
        RXYZ = torch.cov(XYZ.T)
        RXYZ_det = torch.det(RXYZ)
        RXZ = RXYZ[:-1, :-1]
        RXZ_det = torch.det(RXZ)
        for j in range(Na):
            # Delete j-th column and j-th row
            RYZ = RXYZ[:, np.delete(range(Na+1), j)][np.delete(range(Na+1), j), :]
            RYZ_det = torch.det(RYZ)
            RZ = RYZ[:-1, :-1]
            RZ_det = torch.det(RZ)
            CEM[i, j] = 1/2 * torch.log(RYZ_det) - 1/2*torch.log(RZ_det) - 1/2*torch.log(RXYZ_det) + 1/2*torch.log(RXZ_det)
    return CEM

train_LibCG = torch.cat([train_u, train_u[:, :4]**2,
                         train_u[:, [0]]*train_u[:, 1:],
                         train_u[:, [1]]*train_u[:, 2:],
                         train_u[:, [2]]*train_u[:, 3:],
                         train_u[:, [3]]*train_u[:, 4:]], dim=-1)

CEM = cem(train_LibCG, train_u_dot)

CEI = CEM > torch.tensor([0.005, 0.005, 0.005, 0.001, 0.1, 0.1, 0.1]).reshape(-1, 1)



#####################################################
########## Parameter Estimation (Stage 1) ###########
#####################################################

CEI = torch.cat( [ torch.tensor([True]*7).reshape(-1, 1), CEI], dim=-1)
param_matrix = torch.zeros( (CEI.shape[0], CEI.shape[1]) )
for i in range(len(param_matrix)):
    X = train_LibCG[:, CEI[i]]
    y = train_u_dot[:, i]
    param_matrix[i, torch.cat([torch.tensor([True]), CEI[i]])] = torch.inverse(X.T@X)@X.T@y
train_u_dot_pred = torch.cat([torch.ones(Ntrain, 1), train_LibCG], dim=-1) @ param_matrix.T
sigma_hat = torch.sqrt( dt*torch.mean( (train_u_dot - train_u_dot_pred)**2, dim=0 ) )



#####################################################
########## Parameter Estimation (Stage 2) ###########
#####################################################

class RegModel(nn.Module):
    def __init__(self, CEI):
        super().__init__()
        self.CEI = CEI
        self.a = nn.Parameter(torch.randn( torch.sum(CEI[0]) ))
        self.b = nn.Parameter(torch.randn( torch.sum(CEI[1]) ))
        self.c = nn.Parameter(torch.randn( torch.sum(CEI[2]) ))
        self.d = nn.Parameter(torch.randn( torch.sum(CEI[4]) ))
        self.e = nn.Parameter(torch.randn( torch.sum(CEI[5]) ))
        self.f = nn.Parameter(torch.randn( torch.sum(CEI[6]) ))
    def forward(self, t, x):
        param_matrix = torch.zeros(self.CEI.shape[0], self.CEI.shape[1]).to(x.device)
        param_matrix[0, self.CEI[0]] = self.a
        param_matrix[1, self.CEI[1]] = self.b
        param_matrix[2, self.CEI[2]] = self.c
        param_matrix[4, self.CEI[2]] = self.d
        param_matrix[5, self.CEI[2]] = self.e
        param_matrix[6, self.CEI[2]] = self.f
        LibCG = torch.cat([torch.ones(x.shape[0], 1),
                           x, x[:, :4] ** 2,
                           x[:, [0]] * x[:, 1:],
                           x[:, [1]] * x[:, 2:],
                           x[:, [2]] * x[:, 3:],
                           x[:, [3]] * x[:, 4:]], dim=-1)
        x_dot = LibCG @ param_matrix.T
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
    test_u_pred = torchdiffeq.odeint(regmodel, test_u0, test_t[:test_short_steps])
test_u_pred = test_u_pred.permute(1,0,2).reshape(Ntest, -1)

NRMSE(test_u[:, idx_forecast], test_u_pred[:, idx_forecast])



