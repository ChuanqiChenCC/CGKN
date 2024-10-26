# Train Data: PDE Data1; Test Data: PDE Data2
# Forecast Horizon: 12-Months;
# States: TW, TC, TE, WB(external), HW, HC, HE
# Forecast Loss: TW, TC, TE, HW, HC, HE with Path-wise MSE
# FNN for Normalized Data

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


path_abs = r"C:\Users\chenc\CodeProject\CGKN\ENSO"
device = "cuda:1"
torch.manual_seed(0)
np.random.seed(0)

plt.rc("text", usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

#########################################
############# Data Import ###############
#########################################

train_ENSO_mat = sp.io.loadmat(path_abs + "/Data/ENSO_model_data1.mat")
test_ENSO_mat = sp.io.loadmat(path_abs + "/Data/ENSO_model_data2.mat")
dt = 1/360

train_u = np.vstack([np.array(train_ENSO_mat[key]) for key in ["T_W_model", "T_C_model", "T_E_model", "wind_burst_model", "H_W_model", "H_C_model", "H_E_model"]] ).T
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
dim_u = dim_u1 + dim_u2
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

###############################################################
################# Feed-Forward Neural Network #################
###############################################################

class DyNet(nn.Module):
    def __init__(self, dim_u):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_u, 16), nn.ReLU(),
                                 nn.Linear(16, 32), nn.ReLU(),
                                 nn.Linear(32, 64), nn.ReLU(),
                                 nn.Linear(64, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, 16), nn.ReLU(),
                                 nn.Linear(16, dim_u))
    def forward(self, t, x):
        return self.net(x)


################################################
################# Train DyNet  #################
################################################

train_short_steps = int(12/12/dt) # 12m
epochs = 10000
train_batch_size = 1000
dynet = DyNet(dim_u).to(device)
optimizer = torch.optim.Adam(dynet.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
train_loss_history = []
for ep in range(1, epochs+1):
    start_time = time.time()
    optimizer.zero_grad()
    head_indices_short = torch.from_numpy(np.random.choice(Ntrain - train_short_steps + 1, size=train_batch_size, replace=False))
    u_short = torch.stack([train_u_normalized[idx:idx + train_short_steps] for idx in head_indices_short], dim=1).to(device)
    t_short = train_t[:train_short_steps].to(device)
    u_pred_short = torchdiffeq.odeint(dynet, u_short[0], t_short, method="rk4", options={"step_size":0.001})
    loss = nnF.mse_loss(u_short[:, :,  idx_forecast], u_pred_short[:, :, idx_forecast])
    # print(torch.cuda.memory_allocated(device=device) / 1024**2)
    loss.backward()
    optimizer.step()
    scheduler.step()

    train_loss_history.append(loss.item())
    end_time = time.time()
    print(ep, " time:", round(end_time-start_time, 4),
          " loss:", round(loss.item(), 4))

# torch.save(dynet.state_dict(), path_abs + r"/Models/Model_FNN/ENSO_FNN_TH_12Months_Normalized.pt")
# np.save(path_abs + r"/Models/Model_FNN/ENSO_FNN_TH_12Months_Normalized_train_loss_history.npy", train_loss_history)

dynet.load_state_dict(torch.load(path_abs + r"/Models/Model_FNN/ENSO_FNN_TH_12Months_Normalized.pt"))
train_loss_history = np.load(path_abs + r"/Models/Model_FNN/ENSO_FNN_TH_12Months_Normalized_train_loss_history.npy")


##############################################
################# Test dynet #################
##############################################
def NRMSE(u, u_pred):
    return torch.mean(torch.sqrt(torch.mean((u-u_pred)**2, dim=0) / torch.var(u, dim=0))).item()

dynet.cpu()
test_u_normalized = normalizer.encode(test_u)



# Dynamics Prediction
with torch.no_grad():
    test_u_normalized_dot_pred = dynet(None, test_u_normalized)
test_u_dot_pred = normalizer.decode(test_u_normalized_dot_pred)

NRMSE(test_u_dot[:, idx_forecast], test_u_dot_pred[:, idx_forecast])

# State Prediction
test_short_steps = int(12/12/dt)
test_u0_normalized = test_u_normalized[::test_short_steps]
with torch.no_grad():
    test_u_normalized_shortPred = torchdiffeq.odeint(dynet, test_u0_normalized, test_t[:test_short_steps])
test_u_normalized_shortPred = test_u_normalized_shortPred.permute(1, 0, 2).reshape(Ntest, -1)
test_u_shortPred = normalizer.decode(test_u_normalized_shortPred)
nnF.mse_loss(test_u[:, idx_forecast], test_u_shortPred[:, idx_forecast])
NRMSE(test_u[:, idx_forecast], test_u_shortPred[:, idx_forecast])


len( torch.nn.utils.parameters_to_vector(dynet.parameters()))
