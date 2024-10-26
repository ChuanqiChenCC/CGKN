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
device = "cuda:0"
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


######################################################################
################# DyNet with locality and homogeneous  ###############
######################################################################

class DyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 16), nn.ReLU(),
                                 nn.Linear(16, 32), nn.ReLU(),
                                 nn.Linear(32, 64), nn.ReLU(),
                                 nn.Linear(64, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, 16), nn.ReLU(),
                                 nn.Linear(16, 1))
    def forward(self, t, x):
        x_stacked = torch.stack( [x[:, [i-2, i-1, i, (i+1)%40, (i+2)%40]] for i in range(40)], dim=1)
        x_dot = self.net(x_stacked).squeeze(-1)
        return x_dot

################################################
################# Train DyNet  #################
################################################

train_short_steps = int(0.2/dt)
train_batch_size = 400
epochs = 10000
dynet = DyNet().to(device)
optimizer = torch.optim.Adam(dynet.parameters(), 1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
train_loss_history = []
for ep in range(1, epochs+1):
    start_time = time.time()
    optimizer.zero_grad()

    head_indices = np.random.choice(Ntrain-train_short_steps+1, train_batch_size, replace=False)
    u_short = torch.stack([train_u[head_indices+n] for n in range(train_short_steps)]).to(device)
    out = torchdiffeq.odeint(dynet, u_short[0], train_t[:train_short_steps].to(device), method="rk4", options={"step_size":0.001})
    loss = nnF.mse_loss(u_short, out)
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_loss_history.append(loss.item())
    end_time = time.time()
    print("ep:", ep,
          " time:", round(end_time - start_time, 4),
          " loss:", round(loss.item(), 4))

# torch.save(dynet.state_dict(), path_abs + r"/Models/Model_FNN/L94(40)_FNN_sf0u2.pt")
# np.save(path_abs + r"/Models/Model_FNN/L94(40)_FNN_sf0u2_train_loss_history.npy", train_loss_history)


dynet.load_state_dict(torch.load(path_abs + r"/Models/Model_FNN/L94(40)_FNN_sf0u2.pt"))

##############################################
################# Test DyNet #################
##############################################
def NRMSE(u, u_pred):
    return torch.mean(torch.sqrt(torch.mean((u-u_pred)**2, dim=0) / torch.var(u, dim=0))).item()

dynet.cpu()

# Dynamics Prediction
with torch.no_grad():
    test_u_dot_pred = dynet(None, test_u)

NRMSE(test_u_dot, test_u_dot_pred)



# State Prediction
test_short_steps = 200
test_u0 = test_u[::test_short_steps]
with torch.no_grad():
    test_u_shortPred = torchdiffeq.odeint(dynet, test_u0, test_t[:test_short_steps], method="euler", options={"step_size":0.001})
test_u_shortPred = test_u_shortPred.permute(1, 0, 2).reshape(Ntest, -1)
nnF.mse_loss(test_u, test_u_shortPred).item()
NRMSE(test_u, test_u_shortPred)


