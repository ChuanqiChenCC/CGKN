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


#########################################
################# DyNet #################
#########################################

class DyNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 16), nn.ReLU(),
                                 nn.Linear(16, 32), nn.ReLU(),
                                 nn.Linear(32, 64), nn.ReLU(),
                                 nn.Linear(64, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, 16), nn.ReLU(),
                                 nn.Linear(16, dim))
    def forward(self, t, x):
        return self.net(x)

###############################################
################# Train DyNet #################
###############################################

train_short_steps = int(0.1/dt)
train_batch_size = 400
epochs = 10000
dynet = DyNet(3)
optimizer = torch.optim.Adam(dynet.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
train_loss_history = []
for ep in range(1, epochs+1):
    start_time = time.time()
    optimizer.zero_grad()
    head_indices = np.random.choice(Ntrain-train_short_steps+1, train_batch_size, replace=False)
    u_short = torch.stack([train_u[head_indices+n] for n in range(train_short_steps)])
    out = torchdiffeq.odeint(dynet, u_short[0], train_t[:train_short_steps], method="rk4", options={"step_size":0.001})
    loss = nnF.mse_loss(u_short, out)
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_loss_history.append(loss.item())
    end_time = time.time()
    print("ep:", ep,
          " loss:", round(loss.item(), 4),
          " time:", round(end_time-start_time, 4))

# torch.save(dynet.state_dict(), path_abs+"/Models/Model_FNN/PSBSE_FNN.pt")
# np.save(path_abs + "/Models/Model_FNN/PSBSE_FNN_train_loss_history.npy", train_loss_history)

dynet.load_state_dict(torch.load(path_abs+"/Models/Model_FNN/PSBSE_FNN.pt"))


##############################################
################# Test DyNet #################
##############################################
def NRMSE(u, u_pred):
    return torch.mean(torch.sqrt(torch.mean((u-u_pred)**2, dim=0) / torch.var(u, dim=0))).item()

# Dynamics Prediction
with torch.no_grad():
    test_u_dot_pred = dynet(None, test_u)
NRMSE(test_u_dot, test_u_dot_pred)


# State Prediction
test_short_steps = int(0.1/dt)
test_u0 = test_u[::test_short_steps]
with torch.no_grad():
    test_u_shortPred = torchdiffeq.odeint(dynet, test_u0, test_t[:test_short_steps])
test_u_shortPred = test_u_shortPred.permute(1, 0, 2).reshape(Ntest, -1)
NRMSE(test_u, test_u_shortPred)
