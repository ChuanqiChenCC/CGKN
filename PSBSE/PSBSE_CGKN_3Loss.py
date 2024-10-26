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
t = np.linspace(0, Lt, Nt)
u_dot = np.diff(u, axis=0)/dt

# Split data into train and test
u_dot = torch.tensor(u_dot, dtype=torch.float32)
u = torch.tensor(u[:-1], dtype=torch.float32)
t = torch.tensor(t[:-1], dtype=torch.float32)
Nt = len(t)
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


############################################################
################# CGKN: AutoEncoder + CGN  #################
############################################################
class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, 32), nn.ReLU(),
                                     nn.Linear(32, 64), nn.ReLU(),
                                     nn.Linear(64, 64), nn.ReLU(),
                                     nn.Linear(64, 32), nn.ReLU(),
                                     nn.Linear(32, hidden_size))

        self.decoder = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(),
                                     nn.Linear(32, 64), nn.ReLU(),
                                     nn.Linear(64, 64), nn.ReLU(),
                                     nn.Linear(64, 32), nn.ReLU(),
                                     nn.Linear(32, input_size))

class CGN(nn.Module):
    def __init__(self, dim_u1, dim_z):
        super().__init__()
        self.input_size = dim_u1
        self.f1_size = (dim_u1, 1)
        self.g1_size = (dim_u1, dim_z)
        self.f2_size = (dim_z, 1)
        self.g2_size = (dim_z, dim_z)
        self.output_size = np.prod(self.f1_size) + np.prod(self.g1_size) + np.prod(self.f2_size) + np.prod(self.g2_size)
        self.net = nn.Sequential(nn.Linear(self.input_size, 16), nn.ReLU(),
                                 nn.Linear(16, 32), nn.ReLU(),
                                 nn.Linear(32, 64), nn.ReLU(),
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
class CGKN(nn.Module):
    def __init__(self, autoencoder, cgn):
        super().__init__()
        self.autoencoder = autoencoder
        self.cgn = cgn

    def forward(self, t, u_extended):
        # Matrix Form Computation
        dim_u1 = self.cgn.input_size
        u1 = u_extended[:, :dim_u1]
        z = u_extended[:, dim_u1:]
        f1, g1, f2, g2 = self.cgn(u1)
        z = z.unsqueeze(-1)
        u1_dot = f1 + g1@z
        z_dot = f2 + g2@z
        return torch.cat([u1_dot.squeeze(-1), z_dot.squeeze(-1)], dim=-1)

############################################################
################# Train MixModel (Stage1)  #################
############################################################
dim_u1 = 1
dim_u2 = 2
dim_z = 3
dim_f1 = (dim_u1, 1)
dim_g1 = (dim_u1, dim_z)
dim_f2 = (dim_z, 1)
dim_g2 = (dim_z, dim_z)


# Stage1: Train CGKN with loss_forecast + loss_ae
epochs = 500
train_batch_size = 200
train_tensor = torch.utils.data.TensorDataset(train_u, train_u_dot)
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=train_batch_size)
train_num_batches = len(train_loader)
iterations = epochs * train_num_batches
train_loss_forecast_history = []
train_loss_ae_history = []

autoencoder = AutoEncoder(dim_u2, dim_z).to(device)
cgn = CGN(dim_u1, dim_z).to(device)
cgkn = CGKN(autoencoder, cgn).to(device)
optimizer = torch.optim.Adam(cgkn.parameters(), lr=1e-3)
for ep in range(1, epochs+1):
    start_time = time.time()
    train_loss_forecast = 0.
    train_loss_ae = 0.
    for u, u_dot in train_loader:
        u, u_dot = u.to(device), u_dot.to(device)
        optimizer.zero_grad()

        # Encoder
        z = cgkn.autoencoder.encoder(u[:, indices_u2])
        # Decoder
        u2_ae = cgkn.autoencoder.decoder(z)
        loss_ae = nnF.mse_loss(u[:, indices_u2], u2_ae)

        u_extended = torch.cat([u[:, indices_u1], z], dim=-1)
        # Latent One-Step Future State
        u_extended_dot_pred = cgkn(None, u_extended)
        u_extended_pred = u_extended + u_extended_dot_pred*dt # Decoder
        u2_pred = cgkn.autoencoder.decoder(u_extended_pred[:, dim_u1:])

        u_pred = torch.cat([u_extended_pred[:, :dim_u1], u2_pred], dim=-1)
        loss_forecast = nnF.mse_loss(u+u_dot*dt, u_pred)

        loss_total = loss_forecast + loss_ae
        loss_total.backward()
        optimizer.step()

        train_loss_forecast += loss_forecast.item()
        train_loss_ae += loss_ae.item()
    train_loss_forecast /= train_num_batches
    train_loss_ae /= train_num_batches
    train_loss_forecast_history.append(train_loss_forecast)
    train_loss_ae_history.append(train_loss_ae)

    end_time = time.time()
    print(ep, "time:", round(end_time-start_time, 4),
          " loss fore:", round(train_loss_forecast, 4),
          " loss ae:", round(train_loss_ae, 4))

torch.save(cgkn.state_dict(), r"/home/cc/CodeProjects/CGKN/PSBSE/Models/PSBSE_CGKN_dimz_3_stage1.pt")


# # Model Diagnosis in Physical Space
# cgkn.to("cpu")
# with torch.no_grad():
#     test_z = cgkn.autoencoder.encoder(test_u[:, indices_u2])
# test_u_extended = torch.cat([test_u[:, indices_u1], test_z], dim=-1)
# with torch.no_grad():
#     test_u_extended_dot_pred = cgkn(None, test_u_extended)
# test_u_extended_pred = test_u_extended + test_u_extended_dot_pred * dt
# with torch.no_grad():
#     test_u2_pred = cgkn.autoencoder.decoder(test_u_extended_pred[:, dim_u1:])
# test_u_pred = torch.cat([test_u_extended_pred[:, :dim_u1], test_u2_pred], dim=-1)
# test_u_dot_pred = (test_u_pred - test_u)/dt
# nnF.mse_loss(test_u_dot[:, 0], test_u_dot_pred[:, 0])
# nnF.mse_loss(test_u_dot[:, 1], test_u_dot_pred[:, 1])
# nnF.mse_loss(test_u_dot[:, 2], test_u_dot_pred[:, 2])
# cgkn.to(device)


##################################################################
################# Noise Coefficient & CGFilter  #################
##################################################################
cgkn.to("cpu")
# True Train Data in Latent Space
with torch.no_grad():
    train_z = cgkn.autoencoder.encoder(train_u[:, indices_u2])
train_u_extended = torch.cat([train_u[:, indices_u1], train_z], dim=-1)
train_u_extended_dot = torch.diff(train_u_extended, dim=0)/dt
# 1. Prediction in Physical Space
with torch.no_grad():
    train_u_extended_dot_pred = cgkn(None, train_u_extended)
# torch.sqrt( dt*torch.mean( (train_u_extended_dot - train_u_extended_dot_pred[:-1])**2, dim=0 ) )
train_u_extended_pred = train_u_extended + train_u_extended_dot_pred*dt
with torch.no_grad():
    train_u2_pred = cgkn.autoencoder.decoder(train_u_extended_pred[:, dim_u1:])
# 2. Transform to Latent Space
with torch.no_grad():
    train_z2 = cgkn.autoencoder.encoder(train_u2_pred)
train_u_extended_pred2 = torch.cat([train_u_extended_pred[:, :dim_u1], train_z2], dim=-1)
train_u_extended_dot_pred2 = (train_u_extended_pred2 - train_u_extended)/dt
sigma_hat = torch.sqrt( dt*torch.mean( (train_u_extended_dot - train_u_extended_dot_pred2[:-1])**2, dim=0 ) )
cgkn.to(device)


def CGFilter(cgkn, sigma, u1, mu0, R0):
    # u1: (t, x, 1)
    # mu0: (x, 1)
    # R0: (x, x)
    device = u1.device
    Nt = u1.shape[0]
    dim_z = mu0.shape[0]
    mu_posterior = torch.zeros((Nt, dim_z, 1)).to(device)
    R_posterior = torch.zeros((Nt, dim_z, dim_z)).to(device)
    mu_posterior[0] = mu0
    R_posterior[0] = R0
    for n in range(1, Nt):
        f1, g1, f2, g2 = [e.squeeze(0) for e in cgkn.cgn(u1[n-1].T)]
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
    return (mu_posterior, R_posterior)

##########################################################
################# Train cgkn (Stage2)  #################
###########################################################

# Stage 2: Train cgkn with loss_forecast + loss_da + loss_ae
short_steps = int(0.1/dt)
long_steps = int(100/dt)
cut_point = int(5/dt)

epochs = 10000
train_batch_size = 400
train_loss_forecast_history = []
train_loss_da_history = []
train_loss_ae_history = []
# Re-initialize Model
autoencoder = AutoEncoder(dim_u2, dim_z).to(device)
cgn = CGN(dim_u1, dim_z).to(device)
cgkn = CGKN(autoencoder, cgn).to(device)
optimizer = torch.optim.Adam(cgkn.parameters(), lr=1e-3)
for ep in range(1, epochs+1):
    start_time = time.time()
    optimizer.zero_grad()

    head_indices_short = torch.from_numpy(np.random.choice(Ntrain-short_steps+1, size=train_batch_size))
    u_short = torch.stack([train_u[idx:idx + short_steps] for idx in head_indices_short], dim=1).to(device)
    t_short = train_t[:short_steps].to(device)

    # Encoder
    z0_short = cgkn.autoencoder.encoder(u_short[0, :, indices_u2])
    u_extended0_short = torch.cat([u_short[0, :, indices_u1], z0_short], dim=-1)
    # Integration
    u_extended_pred_short = torchdiffeq.odeint(cgkn, u_extended0_short, t_short, method="rk4", options={"step_size":0.001})
    # Decoder
    u2_pred_short = cgkn.autoencoder.decoder(u_extended_pred_short[:, :, dim_u1:])
    u_pred_short = torch.cat( [u_extended_pred_short[:, :, :dim_u1], u2_pred_short], dim=-1)
    loss_forecast = nnF.mse_loss(u_short, u_pred_short)

    head_idx_long = torch.from_numpy( np.random.choice(Ntrain-long_steps+1, size=1) )
    u_long = train_u[head_idx_long:head_idx_long + long_steps].to(device)
    t_long = train_t[head_idx_long:head_idx_long + long_steps].to(device)
    # autoendocer
    z_long = cgkn.autoencoder.encoder(u_long[:, indices_u2])
    u2_ae_long = cgkn.autoencoder.decoder(z_long)
    loss_ae = nnF.mse_loss(u_long[:, indices_u2], u2_ae_long)

    # Filter
    mu_z_pred_long = CGFilter(cgkn, sigma_hat.to(device), u_long[:, indices_u1].unsqueeze(-1), mu0=torch.zeros(dim_z, 1).to(device), R0=0.01*torch.eye(dim_z).to(device))[0].squeeze(-1)
    # Decoder
    mu_pred_long = cgkn.autoencoder.decoder(mu_z_pred_long[cut_point:])
    loss_da = nnF.mse_loss(u_long[cut_point:, indices_u2], mu_pred_long)

    loss_total = loss_forecast + loss_da + loss_ae
    loss_total.backward()
    optimizer.step()

    train_loss_forecast_history.append(loss_forecast.item())
    train_loss_da_history.append(loss_da.item())
    train_loss_ae_history.append(loss_ae.item())

    end_time = time.time()
    print(ep, " time:", round(end_time-start_time, 4),
          " loss fore:", round(loss_forecast.item(), 4),
          " loss da:", round(loss_da.item(),4),
          " loss ae:", round(loss_ae.item(),4))

torch.save(cgkn.state_dict(), r"/home/cc/CodeProjects/CGKN/PSBSE/Models/PSBSE_CGKN_dimz_3_stage2.pt")


#############################################
################# Test cgkn #################
#############################################

cgkn.to("cpu")

# Data Assimilation
with torch.no_grad():
    test_mu_z_pred, test_R_z_pred = CGFilter(cgkn,
                                             sigma_hat,
                                             test_u[:, indices_u1].unsqueeze(-1),
                                             torch.zeros(dim_z, 1),
                                             0.01 * torch.eye(dim_z))
    test_mu_pred = cgkn.autoencoder.decoder(test_mu_z_pred.squeeze(-1))

nnF.mse_loss(test_u[cut_point:, indices_u2], test_mu_pred[cut_point:])


# State Forecast
test_short_steps = int(0.2/dt)
test_num_batches = int(Ntest/test_short_steps)
test_u_shortPred = torch.tensor([])
# for i in range(test_num_batches):
for i in range(10):
    test_u_batch = test_u[i*test_short_steps: (i+1)*test_short_steps]
    with torch.no_grad():
        test_z_batch = cgkn.autoencoder.encoder(test_u_batch[:, indices_u2])
    test_u_extended_batch = torch.cat([test_u_batch[:, indices_u1], test_z_batch], dim=-1)
    with torch.no_grad():
        test_u_extended_pred_batch = torchdiffeq.odeint(cgkn, test_u_extended_batch[[0]], t[:test_short_steps], method="rk4", options={"step_size":0.001}).squeeze(1)
        test_u2_pred_batch = cgkn.autoencoder.decoder(test_u_extended_pred_batch[:, dim_u1:])
    test_u_pred_batch = torch.cat([test_u_extended_pred_batch[:, :dim_u1], test_u2_pred_batch], dim=-1)
    test_u_pred_batch = torch.cat([test_u_pred_batch[:, indices_u1], test_u_pred_batch[:, indices_u2]], dim=-1)
    test_u_shortPred = torch.cat([test_u_shortPred, test_u_pred_batch])

nnF.mse_loss(test_u, test_u_shortPred)


# # Visualization
# fig = plt.figure(figsize=(14, 10))
# axs = fig.subplots(2,1)
# axs[0].plot(test_t, test_u[:, 1], linewidth=3.5, label="True signal", color="blue")
# axs[0].plot(test_t, test_mu_pred[:, 0], linewidth=2.5, label="Posterior mean", color="red")
# axs[0].set_ylabel(r"$x_2$", fontsize=35, rotation=0)
# axs[0].set_title(r"(b) CGK", fontsize=35, rotation=0)
# axs[1].plot(test_t, test_u[:, 3], linewidth=3.5, label="True signal", color="blue")
# axs[1].plot(test_t, test_mu_pred[:, 1], linewidth=2.5, label="Posterior mean", color="red")
# axs[1].set_ylabel(r"$x_4$", fontsize=35, rotation=0)
# axs[1].set_xlabel(r"$t$", fontsize=35)
# for ax in axs.flatten():
#     ax.set_xlim([910, 930])
#     ax.set_xticks(range(910, 930+5, 5))
#     ax.tick_params(labelsize=35, length=8, width=1, direction="in")
#     for spine in ax.spines.values():
#         spine.set_linewidth(1)
# axs[0].set_ylim([-5, 10])
# axs[1].set_ylim([-5, 10])
# axs[0].set_yticks([-4, 0, 4, 8])
# axs[1].set_yticks([-4, 0, 4, 8])
# fig.tight_layout()


len( torch.nn.utils.parameters_to_vector( cgkn.cgn.parameters() ) )
len( torch.nn.utils.parameters_to_vector( cgkn.autoencoder.encoder.parameters() ) )
len( torch.nn.utils.parameters_to_vector( cgkn.autoencoder.decoder.parameters() ) )
