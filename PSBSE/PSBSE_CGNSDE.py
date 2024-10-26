import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torchdiffeq
import time

device = "cpu"
torch.manual_seed(100)
np.random.seed(100)

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
for n in range(Nt-1):
    u[n + 1, 0] = u[n, 0] + (beta_x * u[n,0] + alpha * u[n,0] * u[n,1] + alpha * u[n,1] * u[n,2]) * dt + sigma_x * np.sqrt(dt) * np.random.randn()
    u[n + 1, 1] = u[n, 1] + (beta_y * u[n,1] - alpha * u[n,0] ** 2 + 2 * alpha * u[n,0] * u[n,2]) * dt + sigma_y * np.sqrt(dt) * np.random.randn()
    u[n + 1, 2] = u[n, 2] + (beta_z * u[n,2] - 3 * alpha * u[n,0] * u[n,1]) * dt + sigma_z * np.sqrt(dt) * np.random.randn()

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
Ntrain = int(u.shape[0]*0.8)
Ntest = int(u.shape[0]*0.2)
train_u = u[:Ntrain]
train_u_dot = u_dot[:Ntrain]
train_t = t[:Ntrain]
test_u_dot = u_dot[-Ntest:]
test_u = u[-Ntest:]
test_t = t[-Ntest:]


####################################################
################# CGNN & DynModel  #################
####################################################
# Conditional-Gaussian Neural Network
# (Only taking u1 as input)
class CGNN(nn.Module):
    def __init__(self, input_size, f1_size, g1_size, f2_size, g2_size):
        super().__init__()
        self.input_size = input_size
        self.f1_size = f1_size
        self.g1_size = g1_size
        self.f2_size = f2_size
        self.g2_size = g2_size
        self.output_size = np.prod(f1_size) + np.prod(g1_size) + np.prod(f2_size) + np.prod(g2_size)
        self.net = nn.Sequential(nn.Linear(input_size, 16), nn.ReLU(),
                                 nn.Linear(16, 32), nn.ReLU(),
                                 nn.Linear(32, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU(),
                                 nn.Linear(32, self.output_size))
    def forward(self, x):
        batch_size = x.shape[0]
        out = self.net(x)
        f1 = out[:, :np.prod(self.f1_size)].reshape(batch_size, *self.f1_size)
        g1 = out[:, np.prod(self.f1_size):np.prod(self.f1_size)+np.prod(self.g1_size)].reshape(batch_size, *self.g1_size)
        f2 = out[:, np.prod(self.f1_size)+np.prod(self.g1_size):np.prod(self.f1_size)+np.prod(self.g1_size)+np.prod(self.f2_size)].reshape(batch_size, *self.f2_size)
        g2 = out[:, np.prod(self.f1_size)+np.prod(self.g1_size)+np.prod(self.f2_size):].reshape(batch_size, *self.g2_size)
        return [f1, g1, f2, g2]

    def scale_params(self, factor):
        with torch.no_grad():
            for param in self.parameters():
                param.data *= factor

    def randomize_large_params(self, threshold):
        with torch.no_grad():
            for param in self.parameters():
                mask = (param.data > threshold) | (param.data < -threshold)
                param.data[mask] = torch.empty(param.data[mask].shape).uniform_(-threshold, threshold)


# Neural Stochastic Differential Equation
# (Taking u_extended as input)
class SDEFunc(nn.Module):
    def __init__(self, cgnn):
        super().__init__()
        self.cgnn = cgnn

    def forward(self, t, u):
        # Matrix Form Computation
        dim_u1 = self.cgnn.input_size
        u1 = u[:, :dim_u1]
        u2 = u[:, dim_u1:]
        f1, g1, f2, g2 = self.cgnn(u1)
        u2 = u2.unsqueeze(-1)
        u1_dot = f1 + g1@u2
        u2_dot = f2 + g2@u2
        return torch.cat([u1_dot.squeeze(-1), u2_dot.squeeze(-1)], dim=-1)

############################################################
################# Train MixModel (Stage1)  #################
############################################################
dim_u1 = 1
dim_u2 = 2
dim_f1 = (dim_u1, 1)
dim_g1 = (dim_u1, dim_u2)
dim_f2 = (dim_u2, 1)
dim_g2 = (dim_u2, dim_u2)


# Stage1: Train sdefunc with loss_forecast
epochs = 500
batch_size = 200
train_tensor = torch.utils.data.TensorDataset(train_u, train_u_dot)
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=batch_size)
train_num_batches = len(train_loader)
iterations = epochs * train_num_batches
train_loss_forecast_history = []

cgnn = CGNN(dim_u1, dim_f1, dim_g1, dim_f2, dim_g2).to(device)
sdefunc = SDEFunc(cgnn).to(device)
optimizer = torch.optim.Adam(sdefunc.parameters(), lr=1e-3)
for ep in range(1, epochs+1):
    train_loss_forecast = 0.
    for u, u_dot in train_loader:
        u, u_dot = u.to(device), u_dot.to(device)
        optimizer.zero_grad()
        u_dot_pred = sdefunc(None, u)
        loss_forecast = nnF.mse_loss(u_dot, u_dot_pred)
        loss_forecast.backward()
        optimizer.step()

        train_loss_forecast += loss_forecast.item()
    train_loss_forecast /= train_num_batches
    train_loss_forecast_history.append(train_loss_forecast)
    print(ep, " loss fore:", round(train_loss_forecast, 4))

# torch.save(sdefunc.state_dict(), r"/Users/chuanqichen/File/PythonProject/DANN/PSBSE/Models/XCG_stage1.pt")
# sdefunc.load_state_dict(torch.load(r"/Users/chuanqichen/File/PythonProject/DANN/PSBSE/Models/XCG_stage1.pt"))


##################################################################
################# Noise Coefficient & CGFilter  #################
##################################################################
with torch.no_grad():
    train_u_dot_pred = sdefunc(None, train_u)
sigma_hat = torch.sqrt( dt*torch.mean( (train_u_dot - train_u_dot_pred)**2, dim=0 ) )


def CGFilter(sdefunc, sigma, u1, mu0, R0, cut_point):
    # u1: (t, x, 1)
    # mu0: (x, 1)
    # R0: (x, x)
    device = u1.device
    Nt = u1.shape[0]
    dim_u2 = mu0.shape[0]
    mu_posterior = torch.zeros((Nt, dim_u2, 1)).to(device)
    R_posterior = torch.zeros((Nt, dim_u2, dim_u2)).to(device)
    mu_posterior[0] = mu0
    R_posterior[0] = R0
    for n in range(1, Nt):
        f1, g1, f2, g2 = [e.squeeze(0) for e in sdefunc.cgnn(u1[n-1].T)]
        s1, s2 = torch.diag(sigma[:dim_u1]), torch.diag(sigma[dim_u1:])
        invs1os1 = torch.linalg.inv(s1@s1.T)
        s2os2 = s2@s2.T
        du1 = u1[n] - u1[n-1]
        mu1 = mu0 + (f2+g2@mu0)*dt + (R0@g1.T) @ invs1os1 @ (du1 -(f1+g1@mu0)*dt)
        R1 = R0 + (g2@R0 + R0@g2.T + s2os2 - R0@g1.T@invs1os1@g1@R0)*dt
        mu_posterior[n] = mu1
        R_posterior[n] = R1
        mu0 = mu1
        R0 = R1
    return (mu_posterior[cut_point:], R_posterior[cut_point:])
# num_NaNs = 0
# idx_NaNs = []
# for _ in range(100):
#     print(_)
#     cgnn = CGNN(dim_u1, dim_f1, dim_g1, dim_f2, dim_g2)
#     sdefunc = SDEFunc(cgnn)
#     # sdefunc.cgnn.scale_params(0.05)
#     u1 = test_u[:, :dim_u1].unsqueeze(-1)
#     mu0 = torch.zeros(dim_u2, 1)
#     R0 = 0.01*torch.eye(dim_u2)
#     with torch.no_grad():
#         mumu, RR = CGFilter(sdefunc, sigma_hat, u1, mu0, R0, 0)
#     if torch.any(torch.isnan(mumu)):
#         num_NaNs += 1
#         idx_NaNs.append( torch.nonzero(torch.isnan(mumu.squeeze(-1)))[0,0].item() )



############################################################
################# Train MixModel (Stage2)  #################
############################################################

# Stage 2: Train mixmodel with forcast loss + DA loss
short_steps = int(0.2/dt)
long_steps = int(100/dt)
cut_point = int(5/dt)

epochs = 10000
train_batch_size = 400
train_loss_forecast_history = []
train_loss_da_history = []
train_loss_ae_history = []
# Re-initialize Model
cgnn = CGNN(dim_u1, dim_f1, dim_g1, dim_f2, dim_g2)
sdefunc = SDEFunc(cgnn)
# sdefunc.cgnn.scale_params(0.05)
optimizer = torch.optim.Adam(sdefunc.parameters(), lr=1e-3)
for ep in range(1, epochs+1):
    start_time = time.time()
    optimizer.zero_grad()

    head_indices_short = torch.from_numpy(np.random.choice(Ntrain-short_steps+1, size=train_batch_size))
    u_short = torch.stack([train_u[idx:idx + short_steps] for idx in head_indices_short], dim=1)
    t_short = train_t[:short_steps].to(device)
    u_pred_short = torchdiffeq.odeint(sdefunc, u_short[0], t_short, method="rk4", options={"step_size":0.001})
    loss_forecast = nnF.mse_loss(u_short, u_pred_short)

    head_idx_long = torch.from_numpy( np.random.choice(Ntrain-long_steps+1, size=1) )
    u_long = train_u[head_idx_long:head_idx_long + long_steps].to(device)
    t_long = train_t[head_idx_long:head_idx_long + long_steps].to(device)
    mu_pred_long = CGFilter(sdefunc, sigma_hat, u_long[:, :dim_u1].unsqueeze(-1), mu0=torch.zeros(dim_u2, 1).to(device), R0=0.01*torch.eye(dim_u2).to(device), cut_point=cut_point)[0].squeeze(-1)
    loss_da = nnF.mse_loss(u_long[cut_point:, dim_u1:], mu_pred_long)

    loss_total = loss_forecast + loss_da
    if torch.isnan(loss_total):
        continue
    loss_total.backward()
    optimizer.step()
    train_loss_forecast_history.append(loss_forecast.item())
    train_loss_da_history.append(loss_da.item())

    end_time = time.time()
    print(ep, " time:", round(end_time-start_time, 4),
          " loss fore:", round(loss_forecast.item(), 4),
          " loss da:", round(loss_da.item(),4))

# torch.save(sdefunc.state_dict(), r"/Users/chuanqichen/File/PythonProject/DANN/PSBSE/Models/XCG_stage2.pt")
# sdefunc.load_state_dict(torch.load(r"/Users/chuanqichen/File/PythonProject/DANN/PSBSE/Models/XCG_stage2.pt"))


#################################################
################# Test MixModel #################
#################################################

# Data Assimilation
with torch.no_grad():
    test_mu_pred, test_R_pred = CGFilter(sdefunc, sigma_hat, test_u[:, :dim_u1].unsqueeze(-1).to(device), torch.zeros(dim_u2, 1).to(device), 0.01 * torch.eye(dim_u2).to(device), 0)
test_mu_pred = test_mu_pred.squeeze(-1)

nnF.mse_loss(test_u[cut_point:, dim_u1:], test_mu_pred[cut_point:])



# State Forecast
test_short_steps = int(0.2/dt)
test_num_batches = int(Ntest/test_short_steps)
test_u_shortPred = torch.tensor([])
for i in range(test_num_batches):
    test_u_batch = test_u[i*test_short_steps: (i+1)*test_short_steps]
    with torch.no_grad():
        test_u_pred_batch = torchdiffeq.odeint(sdefunc, test_u_batch[[0]], t[:test_short_steps], method="rk4", options={"step_size":0.001}).squeeze(1)
    test_u_shortPred = torch.cat([test_u_shortPred, test_u_pred_batch])

nnF.mse_loss(test_u, test_u_shortPred)





# # Visualization
# fig = plt.figure(figsize=(14, 10))
# axs = fig.subplots(2,1)
# axs[0].plot(test_t, test_u[:, 1], linewidth=3.5, label="True signal", color="blue")
# axs[0].plot(test_t, test_mu_pred[:, 0], linewidth=2.5, label="Posterior mean", color="red")
# axs[0].set_ylabel(r"$y$", fontsize=35, rotation=0)
# axs[0].set_title(r"(b) CGK with one decoder", fontsize=35, rotation=0)
# axs[1].plot(test_t, test_u[:, 2], linewidth=3.5, label="True signal", color="blue")
# axs[1].plot(test_t, test_mu_pred[:, 1], linewidth=2.5, label="Posterior mean", color="red")
# axs[1].set_ylabel(r"$z$", fontsize=35, rotation=0)
# axs[1].set_xlabel(r"$t$", fontsize=35)
# for ax in axs.flatten():
#     ax.set_xlim([910, 920])
#     ax.tick_params(labelsize=35, length=8, width=1, direction="in")
#     for spine in ax.spines.values():
#         spine.set_linewidth(1)
# axs[0].set_ylim([-5, 2])
# axs[0].set_yticks([-4, -2, 0, 2])
# axs[1].set_ylim([-2.5, 2.5])
# axs[1].set_yticks([-2, 0, 2])
# fig.tight_layout()


len( torch.nn.utils.parameters_to_vector(sdefunc.parameters()) )
