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
Lt = 1000
dt = 0.001
Nt = int(Lt/dt) + 1
t = np.linspace(0, Lt, Nt)
u = np.zeros((Nt, I))

for n in range(Nt-1):
    for i in range(I):
        u_dot = -u[n, i] + u[n,(i+1)%I]*u[n,i-1] - u[n,i-2]*u[n,i-1] + F
        u[n+1, i] = u[n, i] + u_dot*dt + sigma*np.sqrt(dt)*np.random.randn()


# Sub-sampling
u = u[::10]


# u= np.load(r"/Users/chuanqichen/File/PythonProject/CGKN/L96(40)/Data/u.npy")
dt = 0.01
Nt = int(Lt/dt) + 1
t = np.linspace(0, Lt, Nt)
u_dot = np.diff(u, axis=0)/dt

# Split data in to train and test
u_dot = torch.tensor(u_dot, dtype=torch.float32)
u = torch.tensor(u[:-1], dtype=torch.float32)
t = torch.tensor(t[:-1], dtype=torch.float32)

Ntrain = int(u.shape[0] * 0.8)
Ntest = int(u.shape[0] * 0.2)
train_u = u[:Ntrain]
train_u_dot = u_dot[:Ntrain]
train_t = t[:Ntrain]
test_u = u[-Ntest:]
test_u_dot = u_dot[-Ntest:]
test_t = t[-Ntest:]

indices_u1 = np.arange(0, I, 2)
indices_u2 = np.arange(1, I, 2)
dim_u1 = len(indices_u1)
dim_u2 = len(indices_u2)

##################################################################
################# AutoEncoder & CGNN & DynModel  #################
##################################################################

# Design of the CGKN:
# dim_z_local is hyperparameter
# Local & Homo Property are fixed.

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

class CGNN(nn.Module):
    def __init__(self, dim_u1, dim_u2, dim_z_unit):
        super().__init__()
        self.dim_u1 = dim_u1
        self.dim_u2 = dim_u2
        self.dim_z_unit = dim_z_unit

        self.unitNet1 = nn.Sequential(nn.Linear(3, 16), nn.ReLU(),
                                      nn.Linear(16, 32), nn.ReLU(),
                                      nn.Linear(32, 16), nn.ReLU(),
                                      nn.Linear(16, 1 + self.dim_z_unit*3))
        self.unitNet2 = nn.Sequential(nn.Linear(3, 16), nn.ReLU(),
                                      nn.Linear(16, 32), nn.ReLU(),
                                      nn.Linear(32, 16), nn.ReLU(),
                                      nn.Linear(16, self.dim_z_unit + 5*self.dim_z_unit**2))

    def forward(self, x):
        # x is u1
        batch_size = x.shape[0]
        x_local_stacked = torch.stack([x[:, [i - 1, i, (i + 1) % dim_u1]] for i in range(dim_u1)], dim=1)
        out1 = self.unitNet1(x_local_stacked)  # (N, 20, 25) with dim_z_unit = 8
        out2 = self.unitNet2(x_local_stacked)  # (N, 20, 328) with dim_z_unit = 8

        f1 = out1[:, :, :1]
        g1 = torch.zeros(batch_size, self.dim_u1, self.dim_z_unit*self.dim_u2)
        mask1 = torch.arange(dim_u1).unsqueeze(-1)
        mask2 = torch.stack([(torch.arange(3*dim_z_unit)+i*dim_z_unit)%(dim_z_unit*dim_u2) for i in range(-1, dim_u1-1)])
        g1[:, mask1, mask2] = out1[:, :, 1:]

        f2 = out2[:, :, :dim_z_unit].reshape(batch_size, -1, 1)
        g2 = torch.zeros(batch_size, dim_z_unit*dim_u2, dim_z_unit*dim_u2)
        mask1 = torch.arange(dim_z_unit*dim_u2).unsqueeze(-1)
        mask2 = torch.cat([(torch.arange(5*dim_z_unit).repeat(dim_z_unit, 1)+i*dim_z_unit ) % (dim_z_unit*dim_u2) for i in range(-2, dim_u2-2)])
        g2[:, mask1, mask2] = out2[:, :, dim_z_unit:].reshape(batch_size, dim_z_unit*dim_u2, 5*dim_z_unit)
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

class SDEFunc(nn.Module):
    def __init__(self, autoencoder, cgnn):
        super().__init__()
        self.autoencoder = autoencoder
        self.cgnn = cgnn

    def forward(self, t, u_extended):
        # Matrix Form Computation
        dim_u1 = cgnn.dim_u1
        u1 = u_extended[:, :dim_u1]
        z = u_extended[:, dim_u1:]
        f1, g1, f2, g2 = self.cgnn(u1)
        z = z.unsqueeze(-1)
        u1_dot = f1 + g1@z
        z_dot = f2 + g2@z
        return torch.cat([u1_dot.squeeze(-1), z_dot.squeeze(-1)], dim=-1)


############################################################
################# Train MixModel (Stage1)  #################
############################################################
dim_u1 = 20
dim_u2 = 20
dim_z_unit = 5
dim_z = dim_z_unit * dim_u2
dim_f1 = (dim_u1, 1)
dim_g1 = (dim_u1, dim_z)
dim_f2 = (dim_z, 1)
dim_g2 = (dim_z, dim_z)


# Stage1: Train sdefunc with loss_forecast
epochs = 500
batch_size = 200
train_tensor = torch.utils.data.TensorDataset(train_u, train_u_dot)
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=batch_size)
train_num_batches = len(train_loader)
iterations = epochs * train_num_batches
train_loss_forecast_history = []
train_loss_ae_history = []

autoencoder = AutoEncoder(2, dim_z_unit).to(device)
cgnn = CGNN(dim_u1, dim_u2, dim_z_unit).to(device)
sdefunc = SDEFunc(autoencoder, cgnn).to(device)
optimizer = torch.optim.Adam(sdefunc.parameters(), lr=1e-3)
for ep in range(1, epochs+1):
    start_time = time.time()
    train_loss_forecast = 0.
    train_loss_ae = 0.
    for u, u_dot in train_loader:
        u, u_dot = u.to(device), u_dot.to(device)
        optimizer.zero_grad()

        # Encoder
        u2_local_stacked = torch.stack([u[:, indices_u2][:, [i-1, i]] for i in range(dim_u2)], dim=1)
        z_stacked = sdefunc.autoencoder.encoder(u2_local_stacked)
        # Decoder
        u2_local_stacked_ae = sdefunc.autoencoder.decoder(z_stacked)
        loss_ae = nnF.mse_loss(u2_local_stacked, u2_local_stacked_ae)

        z = z_stacked.reshape(batch_size, -1)
        u_extended = torch.cat([u[:, indices_u1], z], dim=-1)
        # Latent One-Step Future State
        u_extended_dot_pred = sdefunc(None, u_extended)
        u_extended_pred = u_extended + u_extended_dot_pred*dt
        # Decoder
        u2_pred_local_stacked = sdefunc.autoencoder.decoder(u_extended_pred[:, dim_u1:].reshape(batch_size, dim_u2, dim_z_unit))
        u2_pred = (u2_pred_local_stacked[:, torch.arange(1, dim_u2+1)%dim_u2, 0] + u2_pred_local_stacked[:, :, 1])/2
        u_pred = torch.stack([u_extended_pred[:, :dim_u1], u2_pred], dim=-1).reshape(batch_size, I)
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
    print(ep, " time:", round(end_time-start_time, 4),
          " loss fore:", round(train_loss_forecast, 4),
          " loss ae:", round(train_loss_ae, 4))


# torch.save(sdefunc.state_dict(), r"/Users/chuanqichen/File/PythonProject/CGKN/L96(40)/Models/L96(40)_CGKN_z5_stage1.pt")
# sdefunc.load_state_dict(torch.load(r"/Users/chuanqichen/File/PythonProject/CGKN/L96(40)/Models/L96(40)_CGKN_z5_stage1.pt"))


# # Model Diagnosis in Physical Space
# train_u2_local_stacked = torch.stack([train_u[:, indices_u2][:, [i-1, i]] for i in range(dim_u2)], dim=1)
# with torch.no_grad():
#     train_z_stacked = sdefunc.autoencoder.encoder(train_u2_local_stacked)
# train_z = train_z_stacked.reshape(Ntrain, dim_z_unit*dim_u2)
# train_u_extended = torch.cat([train_u[:, indices_u1], train_z], dim=-1)
# train_u_extended_dot = torch.diff(train_u_extended, dim=0)/dt
# # 1. Prediction in the Physical Space
# with torch.no_grad():
#     train_u_extended_dot_pred = sdefunc(None, train_u_extended)
# train_u_extended_pred = train_u_extended + train_u_extended_dot_pred*dt
# with torch.no_grad():
#     train_u2_local_stacked_pred = sdefunc.autoencoder.decoder(train_u_extended_pred[:, dim_u1:].reshape(Ntrain, dim_u2, dim_z_unit))
# train_u2_pred = (train_u2_local_stacked_pred[:, torch.arange(1, dim_u2 + 1) % dim_u2, 0] + train_u2_local_stacked_pred[:, :, 1]) / 2
# train_u_pred = torch.stack([train_u_extended_pred[:, :dim_u1], train_u2_pred], dim=-1).reshape(Ntrain, I)
# train_u_dot_pred = (train_u_pred - train_u)/dt
# torch.sqrt(dt*torch.mean( (train_u_dot - train_u_dot_pred)**2, dim=0 ))


##################################################################
################# Noise Coefficient & CGFilter  #################
##################################################################

# True Train Data in Latent Space
train_u2_local_stacked = torch.stack([train_u[:, indices_u2][:, [i-1, i]] for i in range(dim_u2)], dim=1)
with torch.no_grad():
    train_z_stacked = sdefunc.autoencoder.encoder(train_u2_local_stacked)
train_z = train_z_stacked.reshape(Ntrain, dim_z_unit*dim_u2)
train_u_extended = torch.cat([train_u[:, indices_u1], train_z], dim=-1)
train_u_extended_dot = torch.diff(train_u_extended, dim=0)/dt
# 1. Prediction in the Physical Space
with torch.no_grad():
    train_u_extended_dot_pred = sdefunc(None, train_u_extended)
train_u_extended_pred = train_u_extended + train_u_extended_dot_pred*dt
with torch.no_grad():
    train_u2_local_stacked_pred = sdefunc.autoencoder.decoder(train_u_extended_pred[:, dim_u1:].reshape(Ntrain, dim_u2, dim_z_unit))
train_u2_pred = (train_u2_local_stacked_pred[:, torch.arange(1, dim_u2 + 1) % dim_u2, 0] + train_u2_local_stacked_pred[:, :, 1]) / 2
train_u2_local_stacked_pred = torch.stack([train_u2_pred[:, [i - 1, i]] for i in range(dim_u2)], dim=1)
# 2. Transform to Latent Space
with torch.no_grad():
    train_z_stacked2 = sdefunc.autoencoder.encoder(train_u2_local_stacked_pred)
train_z2 = train_z_stacked2.reshape(Ntrain, dim_z_unit*dim_u2)
train_u_extended_pred2 = torch.cat([train_u_extended_pred[:, :dim_u1], train_z2], dim=-1)
train_u_extended_dot_pred2 = (train_u_extended_pred2 - train_u_extended)/dt
sigma_hat = torch.sqrt( dt*torch.mean( (train_u_extended_dot - train_u_extended_dot_pred2[:-1])**2, dim=0 ) )



def CGFilter(sdefunc, sigma, u1, mu0, R0, cut_point):
    # u1: (t, x, 1);  mu0: (x, 1);  R0: (x, x)
    device = u1.device
    Nt = u1.shape[0]
    dim_z = mu0.shape[0]
    mu_posterior = torch.zeros((Nt, dim_z, 1)).to(device)
    R_posterior = torch.zeros((Nt, dim_z, dim_z)).to(device)
    mu_posterior[0] = mu0
    R_posterior[0] = R0
    s1, s2 = torch.diag(sigma[:dim_u1]), torch.diag(sigma[dim_u1:])
    invs1os1 = torch.linalg.inv(s1 @ s1.T)
    s2os2 = s2 @ s2.T
    for n in range(1, Nt):
        f1, g1, f2, g2 = [e.squeeze(0) for e in sdefunc.cgnn(u1[n-1].T)]
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
#     autoencoder = AutoEncoder(2, dim_z_unit)
#     cgnn = CGNN(dim_u1, dim_u2, dim_z_unit)
#     sdefunc = SDEFunc(autoencoder, cgnn)
#
#     u1 = test_u[:, indices_u1].unsqueeze(-1)
#     mu0 = torch.zeros(dim_z, 1)
#     R0 = 0.01*torch.eye(dim_z)
#     with torch.no_grad():
#         mumu, RR = CGFilter(sdefunc, sigma_hat, u1, mu0, R0, 0)
#     if torch.any(torch.isnan(mumu)):
#         num_NaNs += 1
#         idx_NaNs.append( torch.nonzero(torch.isnan(mumu.squeeze(-1)))[0,0].item() )


###########################################################
################# Train sdefunc (Stage2)  #################
###########################################################

# Stage 2: Train sdefunc with loss_forecast + loss_da + loss_ae
short_steps = int(0.1/dt)
long_steps = int(100/dt)
cut_point = int(5/dt)

epochs = 10000
train_batch_size = 400
train_loss_forecast_history = []
train_loss_da_history = []
train_loss_ae_history = []
# # Re-initialize Model
autoencoder = AutoEncoder(2, dim_z_unit)
cgnn = CGNN(dim_u1, dim_u2, dim_z_unit)
sdefunc = SDEFunc(autoencoder, cgnn)
optimizer = torch.optim.Adam(sdefunc.parameters(), lr=1e-3)
for ep in range(1, epochs+1):
    start_time = time.time()
    optimizer.zero_grad()

    head_indices_short = torch.from_numpy(np.random.choice(Ntrain-short_steps+1, size=train_batch_size))
    u_short = torch.stack([train_u[idx:idx + short_steps] for idx in head_indices_short], dim=1)
    t_short = train_t[:short_steps].to(device)
    # Encoder
    u2_t0_local_stacked_short = torch.stack([ u_short[0, :, indices_u2][:, [i - 1, i]] for i in range(dim_u2)], dim=1)
    z0_stacked_short = sdefunc.autoencoder.encoder(u2_t0_local_stacked_short)
    z0_short = z0_stacked_short.reshape(train_batch_size, dim_z_unit * dim_u2)
    u_extended0_short = torch.cat([u_short[0, :, indices_u1], z0_short], dim=-1)
    # Integration
    u_extended_pred_short = torchdiffeq.odeint(sdefunc, u_extended0_short, t_short, method="rk4", options={"step_size":0.001})
    # Decoder
    z_pred_short = u_extended_pred_short[:, :, dim_u1:]
    z_pred_stacked_short = z_pred_short.reshape(short_steps, train_batch_size, dim_u2, dim_z_unit)
    u2_pred_local_stacked_short = sdefunc.autoencoder.decoder(z_pred_stacked_short)
    u2_pred_short = (u2_pred_local_stacked_short[:, :, torch.arange(1, dim_u2+1)%dim_u2, 0] + u2_pred_local_stacked_short[:, :, :, 1])/2
    u_pred_short = torch.stack([u_extended_pred_short[:, :, :dim_u1], u2_pred_short], dim=-1).reshape(short_steps, train_batch_size, I)
    loss_forecast = nnF.mse_loss(u_short, u_pred_short)

    head_idx_long = torch.from_numpy( np.random.choice(Ntrain-long_steps+1, size=1) )
    u_long = train_u[head_idx_long:head_idx_long + long_steps].to(device)
    t_long = train_t[head_idx_long:head_idx_long + long_steps].to(device)
    # autoendocer
    u2_local_stacked_long = torch.stack([u_long[:, indices_u2][:, [i - 1, i]] for i in range(dim_u2)], dim=1)
    z_stacked_long = sdefunc.autoencoder.encoder(u2_local_stacked_long)
    u2_local_stacked_ae_long = sdefunc.autoencoder.decoder(z_stacked_long)
    loss_ae = nnF.mse_loss(u2_local_stacked_long, u2_local_stacked_ae_long)

    # Filter
    mu_z_pred_long = CGFilter(sdefunc, sigma_hat, u_long[:, indices_u1].unsqueeze(-1), mu0=torch.zeros(dim_z, 1).to(device), R0=0.01*torch.eye(dim_z).to(device), cut_point=cut_point)[0].squeeze(-1)
    # Decoder
    mu_z_pred_stacked_long = mu_z_pred_long.reshape(mu_z_pred_long.shape[0], dim_u2, dim_z_unit)
    mu_pred_local_stacked_long = sdefunc.autoencoder.decoder(mu_z_pred_stacked_long)
    mu_pred_long = (mu_pred_local_stacked_long[:, torch.arange(1, dim_u2+1)%dim_u2, 0] + mu_pred_local_stacked_long[:, :, 1])/2
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

# torch.save(sdefunc.state_dict(), r"/Users/chuanqichen/File/PythonProject/CGKN/L96(40)/Models/CGKN_z5_stage2.pt")
sdefunc.load_state_dict(torch.load(r"/Users/chuanqichen/File/PythonProject/CGKN/L96(40)/Models/L96(40)_CGKN_z5_stage2.pt"))


#################################################
################# Test MixModel #################
#################################################

# Data Assimilation
with torch.no_grad():
    test_mu_z_pred, test_R_z_pred = CGFilter(sdefunc,
                                             sigma_hat,
                                             test_u[:, indices_u1].unsqueeze(-1).to(device),
                                             torch.zeros(dim_z, 1).to(device),
                                             0.01 * torch.eye(dim_z).to(device),
                                             0)
test_mu_z_pred = test_mu_z_pred.squeeze(-1)
test_mu_z_pred_stacked = test_mu_z_pred.reshape(Ntest, dim_u2, dim_z_unit)
with torch.no_grad():
    test_mu_pred_local_stacked = sdefunc.autoencoder.decoder(test_mu_z_pred_stacked)
test_mu_pred = (test_mu_pred_local_stacked[:, torch.arange(1, dim_u2+1)%dim_u2,0] + test_mu_pred_local_stacked[:, :, 1])/2
nnF.mse_loss(test_u[cut_point:, indices_u2], test_mu_pred[cut_point:])


# State Forecast
test_short_steps = int(0.1/dt)
test_num_batches = int(Ntest/test_short_steps)
test_u_shortPred = torch.tensor([])
for i in range(test_num_batches):
    test_u_batch = test_u[i*test_short_steps: (i+1)*test_short_steps]
    test_u2_t0_local_stacked_batch = torch.stack([test_u_batch[:1, indices_u2][:, [i - 1, i]] for i in range(dim_u2)], dim=1)
    with torch.no_grad():
        test_z0_stacked_batch = sdefunc.autoencoder.encoder(test_u2_t0_local_stacked_batch)
    test_z0_batch = test_z0_stacked_batch.reshape(1, dim_u2*dim_z_unit)
    test_u0_extended_batch = torch.cat([test_u_batch[[[0]], indices_u1], test_z0_batch], dim=-1)
    with torch.no_grad():
        test_u_extended_pred_batch = torchdiffeq.odeint(sdefunc, test_u0_extended_batch, t[:test_short_steps], method="rk4", options={"step_size":0.001}).squeeze(1)
    test_z_pred_batch = test_u_extended_pred_batch[:, dim_u1:]
    test_z_pred_stacked_batch = test_z_pred_batch.reshape(short_steps, dim_u2, dim_z_unit)
    with torch.no_grad():
        test_u2_pred_local_stacked_batch = sdefunc.autoencoder.decoder(test_z_pred_stacked_batch)
    test_u2_pred_batch = (test_u2_pred_local_stacked_batch[:, torch.arange(1, dim_u2+1)%dim_u2, 0] + test_u2_pred_local_stacked_batch[:, :, 1])/2
    test_u_pred_batch = torch.stack([test_u_extended_pred_batch[:, :dim_u1], test_u2_pred_batch], dim=-1).reshape(short_steps, I)
    test_u_shortPred = torch.cat([test_u_shortPred, test_u_pred_batch])
nnF.mse_loss(test_u, test_u_shortPred)



len( torch.nn.utils.parameters_to_vector( sdefunc.cgnn.parameters() ) )
len( torch.nn.utils.parameters_to_vector( sdefunc.autoencoder.encoder.parameters() ) )
len( torch.nn.utils.parameters_to_vector( sdefunc.autoencoder.decoder.parameters() ) )


