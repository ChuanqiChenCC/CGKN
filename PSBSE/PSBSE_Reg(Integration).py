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

dim_u1 = len(indices_u1)
dim_u2 = len(indices_u2)

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

train_LibCG = torch.stack([train_u[:, 0], train_u[:, 1], train_u[:, 2],train_u[:, 0]**2, train_u[:, 0]*train_u[:, 1], train_u[:, 0]*train_u[:, 2]], dim=-1)
CEM = cem(train_LibCG, train_u_dot)
CEM.numpy().round(3)
CEI = CEM > 0.1

#####################################################
########## Parameter Estimation (Stage 1) ###########
#####################################################
param_matrix = torch.zeros_like(CEM)
for i in range(len(param_matrix)):
    X = train_LibCG[:, CEI[i]]
    y = train_u_dot[:, i]
    param_matrix[i, CEI[i]] = torch.inverse(X.T@X)@X.T@y

train_u_dot_pred = train_LibCG @ param_matrix.T
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
    def forward(self, t, x):
        param_matrix = torch.zeros(self.CEI.shape[0], self.CEI.shape[1]).to(x.device)
        param_matrix[0, self.CEI[0]] = self.a
        param_matrix[1, self.CEI[1]] = self.b
        param_matrix[2, self.CEI[2]] = self.c
        LibCG = torch.stack([x[:, 0], x[:, 1], x[:, 2],
                             x[:, 0]**2, x[:, 0]*x[:, 1], x[:, 0]*x[:, 2]]).T
        x_dot = LibCG @ param_matrix.T
        return x_dot

def CGFilter_Reg(regmodel, sigma, u1, mu0, R0, dt):
    device = u1.device
    Nt = len(u1)
    indices_u1 = np.array([0])
    indices_u2 = np.array([1, 2])
    dim_u1 = len(indices_u1)
    dim_u2 = len(indices_u2)
    a = regmodel.a
    b = regmodel.b
    c = regmodel.c
    s1, s2 = torch.diag(sigma[indices_u1]), torch.diag(sigma[indices_u2])
    invs1os1 = torch.linalg.inv(s1@s1.T)
    s2os2 = s2@s2.T
    mu = torch.zeros(Nt, dim_u2, 1).to(device)
    R = torch.zeros(Nt, dim_u2, dim_u2).to(device)
    mu[0] = mu0
    R[0] = R0
    for n in range(Nt-1):
        f1 = torch.zeros(1, 1).to(device)
        g1 = torch.tensor([[a*u1[n, 0], 0]]).to(device)
        f2 = torch.tensor([ [ b[0]*u1[n, 0]**2], [0] ]).to(device)
        g2 = torch.tensor([ [0, b[1]*u1[n, 0]], [ c*u1[n, 0], 0 ] ]).to(device)
        mu[n+1] = mu[n] + (f2+g2@mu[n])*dt + (R[n]@g1.T) @ invs1os1 @ (u1[n+1] - u1[n] - (f1+g1@mu[n])*dt)
        R[n+1] =  R[n] + (g2@R[n] + R[n]@g2.T + s2os2 - R[n]@g1.T@invs1os1@g1@R[n])*dt
    return mu, R


# Stage 2: Train reg with loss_forecast + loss_da + loss_ae + loss_forecast_z
short_steps = int(0.1/dt)
long_steps = int(50/dt)
cut_point = int(3/dt)

epochs = 10000
train_batch_size = 400
train_loss_forecast_history = []
train_loss_da_history = []
# Re-initialize Model
regmodel = RegModel(CEI).to(device)
optimizer = torch.optim.Adam(regmodel.parameters(), lr=1e-3)
for ep in range(1, epochs+1):
    start_time = time.time()
    optimizer.zero_grad()

    head_indices_short = torch.from_numpy(np.random.choice(Ntrain-short_steps+1, size=train_batch_size, replace=False))
    u_short = torch.stack([train_u[idx:idx + short_steps] for idx in head_indices_short], dim=1).to(device)
    t_short = train_t[:short_steps].to(device)
    u_short_pred = torchdiffeq.odeint(regmodel, u_short[0], t_short, method="rk4", options={"step_size":0.001})
    loss_forecast = nnF.mse_loss(u_short, u_short_pred)

    head_idx_long = torch.from_numpy( np.random.choice(Ntrain-long_steps+1, size=1, replace=False) )
    u_long = train_u[head_idx_long:head_idx_long + long_steps].to(device)
    t_long = train_t[head_idx_long:head_idx_long + long_steps].to(device)
    mu_long_pred = CGFilter_Reg(regmodel, sigma_hat.to(device), u_long[:, indices_u1].unsqueeze(-1), mu0=torch.zeros(dim_u2, 1).to(device), R0=0.01 * torch.eye(dim_u2).to(device), dt=dt)[0].squeeze(-1)
    loss_da = nnF.mse_loss(u_long[cut_point:, indices_u2], mu_long_pred[cut_point:])

    loss_total = loss_forecast + loss_da
    if torch.isnan(loss_total):
        print(ep, "nan")
        continue
    loss_total.backward()
    optimizer.step()
    train_loss_forecast_history.append(loss_forecast.item())
    train_loss_da_history.append(loss_da.item())

    end_time = time.time()
    print(ep, " time:", round(end_time-start_time, 4),
          " loss fore:", round(loss_forecast.item(), 4),
          " loss da:", round(loss_da.item(),4))

# torch.save(regmodel.state_dict(), path_abs + r"/Models/Model_Reg/PSBSE_RegModel.pt")
# np.save(path_abs + r"/Models/Model_Reg/PSBSE_RegModel_train_loss_forecast_history.npy", train_loss_forecast_history)
# np.save(path_abs + r"/Models/Model_Reg/PSBSE_RegModel_train_loss_da_history.npy", train_loss_da_history)

regmodel.load_state_dict(torch.load(path_abs + r"/Models/Model_Reg/PSBSE_RegModel.pt"))


###################################################
########## Test Causal-based Regression ###########
###################################################
def NRMSE(u, u_pred):
    return torch.mean(torch.sqrt(torch.mean((u-u_pred)**2, dim=0) / torch.var(u, dim=0))).item()

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
with torch.no_grad():
    test_mu_pred, test_R_pred = CGFilter_Reg(regmodel, sigma_hat, test_u[:, indices_u1], torch.zeros(dim_u2, 1), 0.01*torch.eye(dim_u2), dt)
test_mu_pred = test_mu_pred.squeeze(-1)

NRMSE(test_u[cut_point:, indices_u2], test_mu_pred[cut_point:])




# np.save(path_abs + r"/Data/PSBSE_Reg_test_mu_pred.npy", test_mu_pred)
# np.save(path_abs + r"/Data/PSBSE_Reg_test_mu_std_pred.npy", test_mu_std_pred)
