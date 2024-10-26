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

############################################################
################# CGKN: AutoEncoder + CGN  #################
############################################################

# Design of the CGKN:
# 1) dim_z_unit is hyperparameter
# 2) Locality & Homogeneity are fixed.

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
    def __init__(self, dim_u1, dim_u2, dim_z_unit):
        super().__init__()
        # 1) Locality is pre-defined by two nearby states at each side of the target state
        self.dim_u1 = dim_u1
        self.dim_u2 = dim_u2
        self.dim_z_unit = dim_z_unit

        # 2) Homogeneity is pre-defined and used with Locality:
        # unitNet1 is for u1_dot and unitNet2 is for u2_dot

        # "Input dim 3" and "3 in Output dim" are pre-defined by locality
        self.unitNet1 = nn.Sequential(nn.Linear(3, 16), nn.ReLU(),
                                      nn.Linear(16, 32), nn.ReLU(),
                                      nn.Linear(32, 16), nn.ReLU(),
                                      nn.Linear(16, 1 + self.dim_z_unit*3))
        # "Input dim 3" and "5 in Output dim" are pre-defined by Locality
        self.unitNet2 = nn.Sequential(nn.Linear(3, 16), nn.ReLU(),
                                      nn.Linear(16, 32), nn.ReLU(),
                                      nn.Linear(32, 16), nn.ReLU(),
                                      nn.Linear(16, self.dim_z_unit + 5*self.dim_z_unit**2))# 5 is pre-defined locality

    def forward(self, x):
        # x is u1
        device = x.device
        batch_size = x.shape[0]
        x_local_stacked = torch.stack([x[:, [i - 1, i, (i + 1) % dim_u1]] for i in range(dim_u1)], dim=1)
        # unitNet1 and unitNet2 are taking the same input (N, 20, 3)
        out1 = self.unitNet1(x_local_stacked)  # Output (N, 20, 19) with dim_z_unit = 6
        out2 = self.unitNet2(x_local_stacked)  # Outpur (N, 20, 186) with dim_z_unit = 6

        f1 = out1[:, :, :1]
        g1 = torch.zeros(batch_size, self.dim_u1, self.dim_z_unit*self.dim_u2).to(device)
        mask1 = torch.arange(dim_u1).unsqueeze(-1)
        mask2 = torch.stack([(torch.arange(3*dim_z_unit)+i*dim_z_unit)%(dim_z_unit*dim_u2) for i in range(-1, dim_u1-1)])
        g1[:, mask1, mask2] = out1[:, :, 1:]

        f2 = out2[:, :, :dim_z_unit].reshape(batch_size, -1, 1)
        g2 = torch.zeros(batch_size, dim_z_unit*dim_u2, dim_z_unit*dim_u2).to(device)
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
class CGKN(nn.Module):
    def __init__(self, autoencoder, cgn):
        super().__init__()
        self.autoencoder = autoencoder
        self.cgn = cgn

    def forward(self, t, u_extended):
        # Matrix Form Computation
        dim_u1 = self.cgn.dim_u1
        u1 = u_extended[:, :dim_u1]
        z = u_extended[:, dim_u1:]
        f1, g1, f2, g2 = self.cgn(u1)
        z = z.unsqueeze(-1)
        u1_dot = f1 + g1@z
        z_dot = f2 + g2@z
        return torch.cat([u1_dot.squeeze(-1), z_dot.squeeze(-1)], dim=-1)

########################################################
################# Train cgkn (Stage1)  #################
########################################################
dim_u1 = len(indices_u1)
dim_u2 = len(indices_u2)
dim_z_unit = 6
dim_z = dim_z_unit * dim_u2


# Stage1: Train cgkn with loss_forecast + loss_ae + loss_forecast_z
epochs = 500
batch_size = 200
train_tensor = torch.utils.data.TensorDataset(train_u, train_u_dot)
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=batch_size)
train_num_batches = len(train_loader)
iterations = epochs * train_num_batches
train_loss_forecast_history = []
train_loss_ae_history = []
train_loss_forecast_z_history = []

autoencoder = AutoEncoder(2, dim_z_unit).to(device) # 2 is pre-defined by Locality
cgn = CGN(dim_u1, dim_u2, dim_z_unit).to(device)
cgkn = CGKN(autoencoder, cgn).to(device)
optimizer = torch.optim.Adam(cgkn.parameters(), lr=1e-3)
for ep in range(1, epochs+1):
    start_time = time.time()
    train_loss_forecast = 0.
    train_loss_ae = 0.
    train_loss_forecast_z = 0.
    for u, u_dot in train_loader:
        u, u_dot = u.to(device), u_dot.to(device)
        optimizer.zero_grad()

        # Encoder
        u2_local_stacked = torch.stack([u[:, indices_u2][:, [i-1, i]] for i in range(dim_u2)], dim=1)
        z_stacked = cgkn.autoencoder.encoder(u2_local_stacked)
        # Decoder
        u2_local_stacked_ae = cgkn.autoencoder.decoder(z_stacked)
        loss_ae = nnF.mse_loss(u2_local_stacked, u2_local_stacked_ae)

        z = z_stacked.reshape(batch_size, -1)
        u_extended = torch.cat([u[:, indices_u1], z], dim=-1)
        # Latent One-Step Future State
        u_extended_dot_pred = cgkn(None, u_extended)
        u_extended_pred = u_extended + u_extended_dot_pred*dt
        # Decoder
        u2_pred_local_stacked = cgkn.autoencoder.decoder(u_extended_pred[:, dim_u1:].reshape(batch_size, dim_u2, dim_z_unit))
        u2_pred = (u2_pred_local_stacked[:, torch.arange(1, dim_u2+1)%dim_u2, 0] + u2_pred_local_stacked[:, :, 1])/2
        u_pred = torch.stack([u_extended_pred[:, :dim_u1], u2_pred], dim=-1).reshape(batch_size, I)
        loss_forecast = nnF.mse_loss(u+u_dot*dt, u_pred)

        # Encoder (Future State)
        u2_next = u[:, indices_u2]+u_dot[:, indices_u2]*dt
        u2_next_local_stacked = torch.stack([u2_next[:, [i-1, i]] for i in range(dim_u2)], dim=1)
        z_stacked = cgkn.autoencoder.encoder(u2_local_stacked)
        z = z_stacked.reshape(batch_size, -1)
        loss_forecast_z = nnF.mse_loss(z, u_extended_pred[:, dim_u1:])

        loss_total = loss_forecast + loss_ae + loss_forecast_z
        loss_total.backward()
        optimizer.step()

        train_loss_forecast += loss_forecast.item()
        train_loss_ae += loss_ae.item()
        train_loss_forecast_z += loss_forecast_z.item()
    train_loss_forecast /= train_num_batches
    train_loss_ae /= train_num_batches
    train_loss_forecast_z /= train_num_batches
    train_loss_forecast_history.append(train_loss_forecast)
    train_loss_ae_history.append(train_loss_ae)
    train_loss_forecast_z_history.append(train_loss_forecast_z)
    end_time = time.time()
    print(ep, " time:", round(end_time-start_time, 4),
          " loss fore:", round(train_loss_forecast, 4),
          " loss ae:", round(train_loss_ae, 4),
          " loss fore z:", round(train_loss_forecast_z, 4))

# torch.save(cgkn.state_dict(), path_abs + r"/Models/Model_CGKN/L94(40)_CGKN_dimzunit7_DA60s_stage1.pt")
# np.save(path_abs + r"/Models/Model_CGKN/L94(40)_CGKN_dimzunit7_DA60s_train_loss_forecast_history_stage1.npy", train_loss_forecast_history)
# np.save(path_abs + r"/Models/Model_CGKN/L94(40)_CGKN_dimzunit7_DA60s_train_loss_ae_history_stage1.npy", train_loss_ae_history)
# np.save(path_abs + r"/codels/Model_CGKN/L94(40)_CGKN_dimzunit7_DA60s_train_loss_forecast_z_history_stage1.npy", train_loss_forecast_z_history)

cgkn.load_state_dict(torch.load(path_abs + r"/Models/Model_CGKN/L94(40)_CGKN_dimzunit6_stage1.pt"))


# # Model Diagnosis in Physical Space
# cgkn.to("cpu")
# train_u2_local_stacked = torch.stack([train_u[:, indices_u2][:, [i-1, i]] for i in range(dim_u2)], dim=1)
# with torch.no_grad():
#     train_z_stacked = cgkn.autoencoder.encoder(train_u2_local_stacked)
# train_z = train_z_stacked.reshape(Ntrain, -1)
# train_u_extended = torch.cat([train_u[:, indices_u1], train_z], dim=-1)
# train_u_extended_dot = torch.diff(train_u_extended, dim=0)/dt
# with torch.no_grad():
#     train_u_extended_dot_pred = cgkn(None, train_u_extended)
# train_u_extended_pred = train_u_extended + train_u_extended_dot_pred*dt
# with torch.no_grad():
#     train_u2_local_stacked_pred = cgkn.autoencoder.decoder(train_u_extended_pred[:, dim_u1:].reshape(Ntrain, dim_u2, dim_z_unit))
# train_u2_pred = (train_u2_local_stacked_pred[:, torch.arange(1, dim_u2 + 1) % dim_u2, 0] + train_u2_local_stacked_pred[:, :, 1]) / 2
# train_u_pred = torch.stack([train_u_extended_pred[:, :dim_u1], train_u2_pred], dim=-1).reshape(Ntrain, I)
# train_u_dot_pred = (train_u_pred - train_u)/dt
# torch.sqrt(dt*torch.mean( (train_u_dot - train_u_dot_pred)**2, dim=0 ))
# cgkn.to(device)

##################################################################
################# Noise Coefficient & CGFilter  #################
##################################################################
cgkn.to("cpu")
train_u2_local_stacked = torch.stack([train_u[:, indices_u2][:, [i-1, i]] for i in range(dim_u2)], dim=1)
with torch.no_grad():
    train_z_stacked = cgkn.autoencoder.encoder(train_u2_local_stacked)
train_z = train_z_stacked.reshape(Ntrain, dim_z_unit*dim_u2)
train_u_extended = torch.cat([train_u[:, indices_u1], train_z], dim=-1)
train_u_extended_dot = torch.diff(train_u_extended, dim=0)/dt
with torch.no_grad():
    train_u_extended_dot_pred = cgkn(None, train_u_extended)
sigma_hat = torch.sqrt( dt*torch.mean( (train_u_extended_dot - train_u_extended_dot_pred[:-1])**2, dim=0 ) )
sigma_hat[dim_u1:] = 1.
cgkn.to(device)


def CGFilter(cgkn, sigma, u1, mu0, R0, dt):
    # u1: (t, x, 1)
    # mu0: (x, 1)
    # R0: (x, x)
    device = u1.device
    Nt = u1.shape[0]
    dim_u1 = u1.shape[1]
    dim_z = mu0.shape[0]
    s1, s2 = torch.diag(sigma[:dim_u1]), torch.diag(sigma[dim_u1:])
    invs1os1 = torch.linalg.inv(s1 @ s1.T)
    s2os2 = s2 @ s2.T
    mu_posterior = torch.zeros((Nt, dim_z, 1)).to(device)
    R_posterior = torch.zeros((Nt, dim_z, dim_z)).to(device)
    mu_posterior[0] = mu0
    R_posterior[0] = R0
    for n in range(1, Nt):
        f1, g1, f2, g2 = [e.squeeze(0) for e in cgkn.cgn(u1[n-1].T)]
        du1 = u1[n] - u1[n-1]
        mu1 = mu0 + (f2+g2@mu0)*dt + (R0@g1.T) @ invs1os1 @ (du1 -(f1+g1@mu0)*dt)
        R1 = R0 + (g2@R0 + R0@g2.T + s2os2 - R0@g1.T@invs1os1@g1@R0)*dt
        mu_posterior[n] = mu1
        R_posterior[n] = R1
        mu0 = mu1
        R0 = R1
    return (mu_posterior, R_posterior)

########################################################
################# Train cgkn (Stage2)  #################
########################################################

# Stage 2: Train cgkn with loss_forecast + loss_da + loss_ae + loss_forecast_z
short_steps = int(0.2/dt)
long_steps = int(60/dt)
cut_point = int(2/dt)

epochs = 10000
train_batch_size = 400
train_loss_forecast_history = []
train_loss_da_history = []
train_loss_ae_history = []
# Re-initialize Model
autoencoder = AutoEncoder(2, dim_z_unit).to(device)
cgn = CGN(dim_u1, dim_u2, dim_z_unit).to(device)
cgkn = CGKN(autoencoder, cgn).to(device)
optimizer = torch.optim.Adam(cgkn.parameters(), lr=1e-3)
for ep in range(1, epochs+1):
    start_time = time.time()
    optimizer.zero_grad()

    head_indices_short = torch.from_numpy(np.random.choice(Ntrain-short_steps+1, size=train_batch_size, replace=False))
    u_short = torch.stack([train_u[idx:idx + short_steps] for idx in head_indices_short], dim=1).to(device)
    t_short = train_t[:short_steps].to(device)
    # Encoder
    u2_local_stacked_short = torch.stack([ u_short[:, :, indices_u2][:, :, [i - 1, i]] for i in range(dim_u2)], dim=2)
    z_stacked_short = cgkn.autoencoder.encoder(u2_local_stacked_short)
    z_short = z_stacked_short.reshape(short_steps, train_batch_size, -1)
    u_extended0_short = torch.cat([u_short[0, :, indices_u1], z_short[0]], dim=-1)

    # Integration
    u_extended_pred_short = torchdiffeq.odeint(cgkn, u_extended0_short, t_short, method="rk4", options={"step_size":0.001})
    z_pred_short = u_extended_pred_short[:, :, dim_u1:]
    loss_forecast_z = nnF.mse_loss(z_short, z_pred_short)

    # Decoder
    z_pred_stacked_short = z_pred_short.reshape(short_steps, train_batch_size, dim_u2, dim_z_unit)
    u2_pred_local_stacked_short = cgkn.autoencoder.decoder(z_pred_stacked_short)
    u2_pred_short = (u2_pred_local_stacked_short[:, :, torch.arange(1, dim_u2+1)%dim_u2, 0] + u2_pred_local_stacked_short[:, :, :, 1])/2
    u_pred_short = torch.stack([u_extended_pred_short[:, :, :dim_u1], u2_pred_short], dim=-1).reshape(short_steps, train_batch_size, I)
    loss_forecast = nnF.mse_loss(u_short, u_pred_short)

    head_idx_long = torch.from_numpy( np.random.choice(Ntrain-long_steps+1, size=1, replace=False) )
    u_long = train_u[head_idx_long:head_idx_long + long_steps].to(device)
    t_long = train_t[head_idx_long:head_idx_long + long_steps].to(device)
    # autoendocer
    u2_local_stacked_long = torch.stack([u_long[:, indices_u2][:, [i - 1, i]] for i in range(dim_u2)], dim=1)
    z_stacked_long = cgkn.autoencoder.encoder(u2_local_stacked_long)
    u2_local_stacked_ae_long = cgkn.autoencoder.decoder(z_stacked_long)
    loss_ae = nnF.mse_loss(u2_local_stacked_long, u2_local_stacked_ae_long)

    # Filter
    mu_z_pred_long = CGFilter(cgkn, sigma_hat.to(device), u_long[:, indices_u1].unsqueeze(-1), mu0=torch.zeros(dim_z, 1).to(device), R0=0.01*torch.eye(dim_z).to(device), dt=dt)[0].squeeze(-1)[cut_point:]
    # Decoder
    mu_z_pred_stacked_long = mu_z_pred_long.reshape(mu_z_pred_long.shape[0], dim_u2, dim_z_unit)
    mu_pred_local_stacked_long = cgkn.autoencoder.decoder(mu_z_pred_stacked_long)
    mu_pred_long = (mu_pred_local_stacked_long[:, torch.arange(1, dim_u2+1)%dim_u2, 0] + mu_pred_local_stacked_long[:, :, 1])/2

    loss_da = nnF.mse_loss(u_long[cut_point:, indices_u2], mu_pred_long)

    loss_total = loss_forecast + loss_da + loss_ae + loss_forecast_z
    if torch.isnan(loss_total):
        print(ep, "nan")
        continue
    loss_total.backward()
    optimizer.step()
    train_loss_forecast_history.append(loss_forecast.item())
    train_loss_da_history.append(loss_da.item())
    train_loss_ae_history.append(loss_ae.item())
    train_loss_forecast_z_history.append(loss_forecast_z.item())

    end_time = time.time()
    print(ep, " time:", round(end_time-start_time, 4),
          " loss fore:", round(loss_forecast.item(), 4),
          " loss da:", round(loss_da.item(),4),
          " loss ae:", round(loss_ae.item(),4),
          " loss fore z:", round(loss_forecast_z.item(), 4))

# torch.save(cgkn.state_dict(), path_abs + r"/Models/Model_CGKN/L94(40)_CGKN_dimzunit7_DA60s_stage2.pt")
# np.save(path_abs + r"/Models/Model_CGKN/L94(40)_CGKN_dimzunit7_DA60s_train_loss_forecast_history_stage2.npy", train_loss_forecast_history)
# np.save(path_abs + r"/Models/Model_CGKN/L94(40)_CGKN_dimzunit7_DA60s_train_loss_da_history_stage2.npy", train_loss_da_history)
# np.save(path_abs + r"/Models/Model_CGKN/L94(40)_CGKN_dimzunit7_DA60s_train_loss_ae_history_stage2.npy", train_loss_ae_history)
# np.save(path_abs + r"/Models/Model_CGKN/L94(40)_CGKN_dimzunit7_DA60s_train_loss_forecast_z_history_stage2.npy", train_loss_forecast_z_history)

cgkn.load_state_dict(torch.load(path_abs + r"/Models/Model_CGKN/L94(40)_CGKN_dimzunit6_stage2.pt"))


########################################################################################
################# Uncertainty Quantification for DA by Residual Analysis ###############
########################################################################################

# DA for Train Data
with torch.no_grad():
    train_mu_z_pred, train_R_z_pred = CGFilter(cgkn,
                                               sigma_hat,
                                               train_u[:, indices_u1].unsqueeze(-1),
                                               torch.zeros(dim_z, 1),
                                               0.01 * torch.eye(dim_z),
                                               dt)
train_mu_z_pred = train_mu_z_pred.squeeze(-1)
train_mu_z_pred_stacked = train_mu_z_pred.reshape(Ntrain, dim_u2, dim_z_unit)
with torch.no_grad():
    train_mu_pred_local_stacked = cgkn.autoencoder.decoder(train_mu_z_pred_stacked)
train_mu_pred = (train_mu_pred_local_stacked[:, torch.arange(1, dim_u2+1)%dim_u2,0] + train_mu_pred_local_stacked[:, :, 1])/2

nnF.mse_loss(train_u[cut_point:, indices_u2], train_mu_pred[cut_point:])


# Target Variable: Residual (std of posterior mean)
train_mu_std = torch.abs(train_u[cut_point:, indices_u2] - train_mu_pred[cut_point:])

class UncertaintyNet(nn.Module):
    def __init__(self, dim_u1, dim_u2):
        super().__init__()
        self. net = nn.Sequential(nn.Linear(dim_u1, 16), nn.ReLU(),
                                  nn.Linear(16, 32), nn.ReLU(),
                                  nn.Linear(32, 32), nn.ReLU(),
                                  nn.Linear(32, 32), nn.ReLU(),
                                  nn.Linear(32, 16), nn.ReLU(),
                                  nn.Linear(16, dim_u2))

    def forward(self, x):
        # x is u1
        out = self.net(x)
        # out is std of posterior mu
        return out

epochs = 500
train_batch_size = 200
train_tensor = torch.utils.data.TensorDataset(train_u[cut_point:, indices_u1], train_mu_std)
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=train_batch_size)
train_num_batches = len(train_loader)
iterations = epochs * train_num_batches
train_loss_uncertainty_history = []

uncertainty_net = UncertaintyNet(dim_u1, dim_u2)
optimizer = torch.optim.Adam(uncertainty_net.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
for ep in range(1, epochs+1):
    start_time = time.time()
    train_loss_uncertainty = 0.
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = uncertainty_net(x)
        loss = nnF.mse_loss(y, out)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss_uncertainty += loss.item()
    train_loss_uncertainty /= train_num_batches
    train_loss_uncertainty_history.append(train_loss_uncertainty)
    end_time = time.time()
    print(ep," time:", round(end_time - start_time, 4),
          " loss uncertainty:", round(train_loss_uncertainty, 4))



#############################################
################# Test cgkn #################
#############################################
def NRMSE(u, u_pred):
    return torch.mean(torch.sqrt(torch.mean((u-u_pred)**2, dim=0) / torch.var(u, dim=0))).item()

cgkn.to("cpu")

# CGKN for Dynamics Prediction
test_u2_0_local_stacked = torch.stack([ test_u[:, indices_u2][:, [i - 1, i]] for i in range(dim_u2)], dim=-2)
test_z_0_stacked = cgkn.autoencoder.encoder(test_u2_0_local_stacked)
test_z_0 = test_z_0_stacked.reshape(test_z_0_stacked.shape[0], -1)
test_u_extended0 = torch.cat([test_u[:, indices_u1], test_z_0], dim=-1)
with torch.no_grad():
    test_u_extended_pred = torchdiffeq.odeint(cgkn, test_u_extended0, train_t[:2], method="rk4", options={"step_size":0.001})[-1]
test_z_pred = test_u_extended_pred[:, dim_u1:]
test_z_pred_stacked = test_z_pred.reshape(test_z_pred.shape[0], dim_u2, dim_z_unit)
with torch.no_grad():
    test_u2_pred_local_stacked = cgkn.autoencoder.decoder(test_z_pred_stacked)
test_u2_pred = (test_u2_pred_local_stacked[:, torch.arange(1, dim_u2+1)%dim_u2, 0] + test_u2_pred_local_stacked[:, :, 1])/2
test_u_pred = torch.stack([test_u_extended_pred[:, :dim_u1], test_u2_pred], dim=-1).reshape(-1, I)
test_u_dot_pred = (test_u_pred - test_u)/dt
NRMSE(test_u_dot, test_u_dot_pred)



# CGKN for State Prediction
test_short_steps = 20
test_u0 = test_u[::test_short_steps]
test_u2_0_local_stacked = torch.stack([ test_u0[:, indices_u2][:, [i - 1, i]] for i in range(dim_u2)], dim=-2)
test_z_0_stacked = cgkn.autoencoder.encoder(test_u2_0_local_stacked)
test_z_0 = test_z_0_stacked.reshape(test_z_0_stacked.shape[0], -1)
test_u_extended0 = torch.cat([test_u0[:, indices_u1], test_z_0], dim=-1)
with torch.no_grad():
    test_u_extended_shortPred = torchdiffeq.odeint(cgkn, test_u_extended0, t[:test_short_steps], method="rk4", options={"step_size":0.001})
test_z_shortPred = test_u_extended_shortPred[:, :, dim_u1:]
test_z_shortPred_stacked = test_z_shortPred.reshape(test_z_shortPred.shape[0], test_z_shortPred.shape[1], dim_u2, dim_z_unit)
with torch.no_grad():
    test_u2_shortPred_local_stacked = cgkn.autoencoder.decoder(test_z_shortPred_stacked)
test_u2_shortPred = (test_u2_shortPred_local_stacked[:, :, torch.arange(1, dim_u2+1)%dim_u2, 0] + test_u2_shortPred_local_stacked[:, :, :, 1])/2
test_u_shortPred = torch.stack([test_u_extended_shortPred[:, :, :dim_u1], test_u2_shortPred], dim=-1).reshape(test_short_steps, -1, I)
test_u_shortPred = test_u_shortPred.permute(1, 0, 2).reshape(-1, I)[:Ntest]
nnF.mse_loss(test_u, test_u_shortPred)
NRMSE(test_u, test_u_shortPred)


# CGKN for Lead Forecast
si = int(100/dt) # 400u
ei = int(150/dt) # 450u
test_lead_steps = int(0.2/dt)
test_u0 = test_u[si:ei - test_lead_steps]
test_u2_0_local_stacked = torch.stack([ test_u0[:, indices_u2][:, [i - 1, i]] for i in range(dim_u2)], dim=-2)
test_z_0_stacked = cgkn.autoencoder.encoder(test_u2_0_local_stacked)
test_z_0 = test_z_0_stacked.reshape(test_z_0_stacked.shape[0], -1)
test_u_extended0 = torch.cat([test_u0 [:, indices_u1], test_z_0], dim=-1)
with torch.no_grad():
    test_u_extended_leadPred = torchdiffeq.odeint(cgkn, test_u_extended0, t[:test_lead_steps], method="rk4", options={"step_size":0.001})[-1]
test_z_leadPred = test_u_extended_leadPred[:, dim_u1:]
test_z_leadPred_stacked = test_z_leadPred.reshape(test_z_leadPred.shape[0], dim_u2, dim_z_unit)
with torch.no_grad():
    test_u2_leadPred_local_stacked = cgkn.autoencoder.decoder(test_z_leadPred_stacked)
test_u2_leadPred = (test_u2_leadPred_local_stacked[:, torch.arange(1, dim_u2+1)%dim_u2, 0] + test_u2_leadPred_local_stacked[:, :, 1])/2
test_u_leadPred = torch.stack([test_u_extended_leadPred[:, :dim_u1], test_u2_leadPred], dim=-1).reshape(ei-si-test_lead_steps, I)

# np.save(path_abs + r"/Data/L96(40)_CGKN_dimzunit6_test_u_leadPred0u2(400uTo450u).npy", test_u_leadPred)



# CGKN for Data Assimilation
with torch.no_grad():
    test_mu_z_pred, test_R_z_pred = CGFilter(cgkn,
                                             sigma_hat,
                                             test_u[:, indices_u1].unsqueeze(-1),
                                             torch.zeros(dim_z, 1),
                                             0.01 * torch.eye(dim_z),
                                             dt)
test_mu_z_pred = test_mu_z_pred.squeeze(-1)
test_mu_z_pred_stacked = test_mu_z_pred.reshape(Ntest, dim_u2, dim_z_unit)
with torch.no_grad():
    test_mu_pred_local_stacked = cgkn.autoencoder.decoder(test_mu_z_pred_stacked)
test_mu_pred = (test_mu_pred_local_stacked[:, torch.arange(1, dim_u2+1)%dim_u2,0] + test_mu_pred_local_stacked[:, :, 1])/2
NRMSE(test_u[cut_point:, indices_u2], test_mu_pred[cut_point:])

# UncertaintyNet for Uncertainty Quantification
with torch.no_grad():
    test_mu_std_pred = uncertainty_net(test_u[:, indices_u1])

# np.save(path_abs + r"/Data/L96(40)_CGKN_dimzunit6_test_mu_pred.npy", test_mu_pred)
# np.save(path_abs + r"/Data/L96(40)_CGKN_dimzunit6_test_mu_std_pred.npy", test_mu_std_pred)



# CGKN: Number of Parameters
len( torch.nn.utils.parameters_to_vector( cgkn.cgn.parameters() ) )
len( torch.nn.utils.parameters_to_vector( cgkn.cgn.unitNet1.parameters() ) )
len( torch.nn.utils.parameters_to_vector( cgkn.cgn.unitNet2.parameters() ) )
len( torch.nn.utils.parameters_to_vector( cgkn.autoencoder.encoder.parameters() ) )
len( torch.nn.utils.parameters_to_vector( cgkn.autoencoder.decoder.parameters() ) )

