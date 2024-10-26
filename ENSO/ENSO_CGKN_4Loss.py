# Train Data: PDE Data1; Test Data: PDE Data2
# Observable Variable: TW,TC,TE, WB(external); Unobservable: HW, HC, HE
# Forecast Horizon: 1-day; DA Horizon: 40 Years with 4 Years be cut-off
# Target Loss (4): AE Loss + Forecast Loss ++ Forecast Loss z + DA Loss
# Forecast Loss: TW, TC, TE, HW, HC, HE with Path-wise MSE in Original Space
# Forecast Loss z: z with Path-wise MSE in Latent Space
# CGKN for Normalized Data with dimz = 10

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

path_abs = r"C:\Users\Patron\CodeProjects\CGKN\ENSO"
device = "cpu"
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

############################################################
################# CGKN: AutoEncoder + CGN  #################
############################################################

class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, 32), nn.ReLU(),
                                     nn.Linear(32, 64), nn.ReLU(),
                                     nn.Linear(64, 64), nn.ReLU(),
                                     nn.Linear(64, 64), nn.ReLU(),
                                     nn.Linear(64, 32), nn.ReLU(),
                                     nn.Linear(32, hidden_size))

        self.decoder = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(),
                                     nn.Linear(32, 64), nn.ReLU(),
                                     nn.Linear(64, 64), nn.ReLU(),
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
                                 nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, 16), nn.ReLU(),
                                 nn.Linear(16, self.output_size))
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

########################################################
################# Train cgkn (Stage1)  #################
########################################################
dim_u1 = len(indices_u1)
dim_u2 = len(indices_u2)
dim_z = 10

# Stage1: Train cgkn with loss_forecast + loss_ae + loss_forecast_z
epochs = 500
train_batch_size = 1000
train_tensor = torch.utils.data.TensorDataset(train_u_normalized, train_u_dot_normalized)
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=train_batch_size)
train_num_batches = len(train_loader)
iterations = epochs * train_num_batches
train_loss_forecast_history = []
train_loss_ae_history = []
train_loss_forecast_z_history = []

autoencoder = AutoEncoder(dim_u2, dim_z).to(device)
cgn = CGN(dim_u1, dim_z).to(device)
cgkn = CGKN(autoencoder, cgn).to(device)
optimizer = torch.optim.Adam(cgkn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
for ep in range(1, epochs+1):
    start_time = time.time()

    train_loss_forecast = 0.
    train_loss_ae = 0.
    train_loss_forecast_z = 0.
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
        u_extended_pred = u_extended + u_extended_dot_pred*dt
        # Decoder
        u2_pred = cgkn.autoencoder.decoder(u_extended_pred[:, dim_u1:])
        u_pred = torch.cat([u_extended_pred[:, :dim_u1], u2_pred], dim=-1)
        loss_forecast = nnF.mse_loss(u+u_dot*dt, u_pred)

        # Encoder (Future State)
        z = cgkn.autoencoder.encoder(u[:, indices_u2]+u_dot[:, indices_u2]*dt)
        loss_forecast_z = nnF.mse_loss(z, u_extended_pred[:, dim_u1:])

        loss_total = loss_forecast + loss_ae + loss_forecast_z
        loss_total.backward()
        optimizer.step()
        scheduler.step()

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
    print(ep, "time:", round(end_time - start_time, 4),
          " loss fore:", round(train_loss_forecast, 4),
          " loss ae:", round(train_loss_ae, 4),
          " loss fore z:", round(train_loss_forecast_z, 4))

torch.save(cgkn.state_dict(), path_abs + r"/Models/Model_CGKN/ENSO_CGKN_TH_1day_4Loss_Normalized_dimz10_stage1.pt")
np.save(path_abs + r"/Models/Model_CGKN/ENSO_CGKN_TH_1day_4Loss_Normalized_dimz10_train_loss_forecast_history_stage1.npy", train_loss_forecast_history)
np.save(path_abs + r"/Models/Model_CGKN/ENSO_CGKN_TH_1day_4Loss_Normalized_dimz10_train_loss_ae_history_stage1.npy", train_loss_ae_history)
np.save(path_abs + r"/Models/Model_CGKN/ENSO_CGKN_TH_1day_4Loss_Normalized_dimz10_train_loss_forecast_z_history_stage1.npy", train_loss_forecast_z_history)

# # Model Diagnosis in Physical Space
# cgkn.to("cpu")
# with torch.no_grad():
#     train_z = cgkn.autoencoder.encoder(train_u_normalized[:, indices_u2])
# train_u_normalized_extended = torch.cat([train_u_normalized[:, indices_u1], train_z], dim=-1)
# with torch.no_grad():
#     train_u_normalized_extended_dot_pred = cgkn(None, train_u_normalized_extended)
# train_u_normalized_extended_pred = train_u_normalized_extended + train_u_normalized_extended_dot_pred * dt
# with torch.no_grad():
#     train_u2_normalized_pred = cgkn.autoencoder.decoder(train_u_normalized_extended_pred[:, dim_u1:])
# train_u_normalized_pred = torch.cat([train_u_normalized_extended_pred[:, :dim_u1], train_u2_normalized_pred], dim=-1)
# train_u_pred = normalizer.decode(train_u_normalized_pred)
# train_u_dot_pred = (train_u_pred - train_u)/dt
# nnF.mse_loss(train_u_dot[:, 0], train_u_dot_pred[:, 0])
# nnF.mse_loss(train_u_dot[:, 1], train_u_dot_pred[:, 1])
# nnF.mse_loss(train_u_dot[:, 2], train_u_dot_pred[:, 2])
# nnF.mse_loss(train_u_dot[:, 3], train_u_dot_pred[:, 3])
# nnF.mse_loss(train_u_dot[:, 4], train_u_dot_pred[:, 4])
# nnF.mse_loss(train_u_dot[:, 5], train_u_dot_pred[:, 5])
# nnF.mse_loss(train_u_dot[:, 6], train_u_dot_pred[:, 6])
# cgkn.to(device)


#################################################################
################# Noise Coefficient & CGFilter  #################
#################################################################
cgkn.to("cpu")
with torch.no_grad():
    train_z = cgkn.autoencoder.encoder(train_u_normalized[:, indices_u2])
train_u_normalized_extended = torch.cat([train_u_normalized[:, indices_u1], train_z], dim=-1)
train_u_normalized_extended_dot = torch.diff(train_u_normalized_extended, dim=0)/dt
with torch.no_grad():
    train_u_normalized_extended_dot_pred = cgkn(None, train_u_normalized_extended)
sigma_hat = torch.sqrt( dt*torch.mean( (train_u_normalized_extended_dot - train_u_normalized_extended_dot_pred[:-1])**2, dim=0 ) )
sigma_hat[dim_u1:] = 0.2  # sigma2 is set manually
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
    invs1os1 = torch.linalg.inv(s1@s1.T)
    s2os2 = s2@s2.T
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
short_steps =  2 # 1 day
long_steps = int(40/dt) # 40y
cut_point = int(4/dt) # 4y
idx_forecast = [0, 1, 2, 4, 5, 6]

epochs = 10000
train_batch_size = 1000
train_loss_forecast_history = []
train_loss_da_history = []
train_loss_ae_history = []
train_loss_forecast_z_history = []
# Re-initialize Model
autoencoder = AutoEncoder(dim_u2, dim_z).to(device)
cgn = CGN(dim_u1, dim_z).to(device)
cgkn = CGKN(autoencoder, cgn).to(device)
optimizer = torch.optim.Adam(cgkn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
for ep in range(1, epochs+1):
    start_time = time.time()
    optimizer.zero_grad()

    head_indices_short = torch.from_numpy(np.random.choice(Ntrain-short_steps+1, size=train_batch_size, replace=False))
    u_short = torch.stack([train_u_normalized[idx:idx + short_steps] for idx in head_indices_short], dim=1).to(device)
    t_short = train_t[:short_steps].to(device)
    # Encoder
    z_short = cgkn.autoencoder.encoder(u_short[:, :, indices_u2])
    z0_short = z_short[0]
    u_extended0_short = torch.cat([u_short[0, :, indices_u1], z0_short], dim=-1)

    # Integration
    u_extended_pred_short = torchdiffeq.odeint(cgkn, u_extended0_short, t_short, method="rk4", options={"step_size":1/720})
    loss_forecast_z = nnF.mse_loss(z_short, u_extended_pred_short[:, :, dim_u1:])
    # Decoder
    u2_pred_short = cgkn.autoencoder.decoder(u_extended_pred_short[:, :, dim_u1:])
    u_pred_short = torch.cat([u_extended_pred_short[:, :, :dim_u1], u2_pred_short], dim=-1)
    loss_forecast = nnF.mse_loss(u_short[:, :,  idx_forecast], u_pred_short[:, :, idx_forecast])

    head_idx_long = torch.from_numpy( np.random.choice(Ntrain-long_steps+1, size=1, replace=False) )
    u_long = train_u_normalized[head_idx_long:head_idx_long + long_steps].to(device)
    t_long = train_t[head_idx_long:head_idx_long + long_steps].to(device)
    # autoendocer
    z_long = cgkn.autoencoder.encoder(u_long[:, indices_u2])
    u2_ae_long = cgkn.autoencoder.decoder(z_long)
    loss_ae = nnF.mse_loss(u_long[:, indices_u2], u2_ae_long)

    # Filter
    mu_z_pred_long = CGFilter(cgkn, sigma_hat.to(device), u_long[:, indices_u1].unsqueeze(-1), mu0=torch.zeros(dim_z, 1).to(device), R0=0.01*torch.eye(dim_z).to(device), dt=dt)[0].squeeze(-1)
    # Decoder
    mu_pred_long = cgkn.autoencoder.decoder(mu_z_pred_long[cut_point:])
    loss_da = nnF.mse_loss(u_long[cut_point:, indices_u2], mu_pred_long)

    loss_total = loss_forecast + loss_da + loss_ae + loss_forecast_z
    if torch.isnan(loss_total):
        print(ep, "nan")
        continue
    loss_total.backward()
    optimizer.step()
    scheduler.step()

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


torch.save(cgkn.state_dict(), path_abs + r"/Models/Model_CGKN/ENSO_CGKN_TH_1day_4Loss_Normalized_dimz10_stage2.pt")
np.save(path_abs + r"/Models/Model_CGKN/ENSO_CGKN_TH_1day_4Loss_Normalized_dimz10_train_loss_forecast_history_stage2.npy", train_loss_forecast_history)
np.save(path_abs + r"/Models/Model_CGKN/ENSO_CGKN_TH_1day_4Loss_Normalized_dimz10_train_loss_ae_history_stage2.npy", train_loss_ae_history)
np.save(path_abs + r"/Models/Model_CGKN/ENSO_CGKN_TH_1day_4Loss_Normalized_dimz10_train_loss_forecast_z_history_stage2.npy", train_loss_forecast_z_history)
np.save(path_abs + r"/Models/Model_CGKN/ENSO_CGKN_TH_1day_4Loss_Normalized_dimz10_train_loss_da_history_stage2.npy", train_loss_da_history)

########################################################################################
################# Uncertainty Quanitficaiton for DA by Residual Analysis ###############
########################################################################################

# DA for Train Data
with torch.no_grad():
    train_mu_z_pred, train_R_z_pred = CGFilter(cgkn,
                                               sigma_hat,
                                               train_u_normalized[:, indices_u1].unsqueeze(-1),
                                               torch.zeros(dim_z, 1),
                                               0.01 * torch.eye(dim_z),
                                               dt)
    train_mu_normalized_pred = cgkn.autoencoder.decoder(train_mu_z_pred.squeeze(-1))

train_mu_pred = (normalizer.std[dim_u1:] + normalizer.eps)*train_mu_normalized_pred + normalizer.mean[dim_u1:]

nnF.mse_loss(train_u[cut_point:, indices_u2], train_mu_pred[cut_point:])
torch.corrcoef( torch.cat([train_u[cut_point:, [4]], train_mu_pred[cut_point:, [0] ]], dim=-1).T )

# Target Variable: Residual (std of posterior mean)
train_mu_std = torch.abs(train_u[cut_point:, indices_u2] - train_mu_pred[cut_point:])

class UncertaintyNet(nn.Module):
    def __init__(self, dim_u1, dim_u2):
        super().__init__()
        self. net = nn.Sequential(nn.Linear(dim_u1, 16), nn.ReLU(),
                                  nn.Linear(16, 32), nn.ReLU(),
                                  nn.Linear(32, 32), nn.ReLU(),
                                  nn.Linear(32, 16), nn.ReLU(),
                                  nn.Linear(16, dim_u2))

    def forward(self, x):
        # x is u1
        out = self.net(x)
        # out is std of posterior mu
        return out

epochs = 500
train_batch_size = 400
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

# torch.save(uncertainty_net.state_dict(), path_abs + r"/Models/Model_CGKN/ENSO_UQNet_TH_6Months_3Loss_Normalized_dimz15.pt")
# np.save(path_abs + r"/Models/Model_CGKN/ENSO_UQNet_TH_6Months_3Loss_Normalized_dimz15_train_loss_uncertainty_history.npy", train_loss_uncertainty_history)

uncertainty_net.load_state_dict(torch.load(path_abs + r"/Models/Model_CGKN/ENSO_UQNet_TH_6Months_3Loss_Normalized_dimz15.pt"))


#############################################
################# Test cgkn #################
#############################################
cgkn.to("cpu")
test_u_normalized = normalizer.encode(test_u)


# CGKN for State Prediction
test_short_steps = int(0.3/dt)
test_u0 = test_u_normalized[::test_short_steps]
with torch.no_grad():
    test_z0 = cgkn.autoencoder.encoder(test_u0[:, indices_u2])
test_u_extended0 = torch.cat([test_u0[:, indices_u1], test_z0], dim=-1)
with torch.no_grad():
    test_u_extended_shortPred = torchdiffeq.odeint(cgkn, test_u_extended0, test_t[:test_short_steps], method="rk4", options={"step_size":1/720})
    test_u2_shortPred = cgkn.autoencoder.decoder(test_u_extended_shortPred[:, :, dim_u1:])
test_u_normalized_shortPred = torch.cat([test_u_extended_shortPred[:, :, :dim_u1], test_u2_shortPred], dim=-1)
test_u_normalized_shortPred = test_u_normalized_shortPred.permute(1, 0, 2).reshape(-1, dim_u1+dim_u2)[:Ntest]
test_u_shortPred = normalizer.decode(test_u_normalized_shortPred)

nnF.mse_loss(test_u[:, idx_forecast], test_u_shortPred[:, idx_forecast])
# np.save(path_abs + r"/Models/Model_CGKN/6MsRelayForecast.npy", test_u_shortPred)



# CGKN for Lead Forecast
test_lead_steps = int(1/dt)
si = int(1000/dt)
ei = int(1100/dt)
test_u_normalized0 = test_u_normalized[si:ei-test_lead_steps]
with torch.no_grad():
    test_z0 = cgkn.autoencoder.encoder(test_u_normalized0[:, indices_u2])
test_u_extended0 = torch.cat([test_u_normalized0[:, indices_u1], test_z0], dim=-1)
with torch.no_grad():
    test_u_extended_leadPred = torchdiffeq.odeint(cgkn, test_u_extended0, test_t[:test_lead_steps], method="rk4", options={"step_size":1/720})[-1]
    test_u2_normalized_leadPred = cgkn.autoencoder.decoder(test_u_extended_leadPred[:, dim_u1:])
test_u_normalized_leadPred = torch.cat([test_u_extended_leadPred[:, :dim_u1], test_u2_normalized_leadPred], dim=-1)
test_u_leadPred = normalizer.decode(test_u_normalized_leadPred)

1 - torch.sqrt(nnF.mse_loss(test_u[si+test_lead_steps:ei, 2], test_u_leadPred[:, 2]) / torch.var(test_u[si+test_lead_steps:ei, 2]))
torch.corrcoef( torch.stack([test_u[si+test_lead_steps:ei, 2], test_u_leadPred[:, 2]]) )

# np.save(path_abs + r"/Models/Model_CGKN/6MsLeadForecast(1000Yto1100Y).npy", test_u_leadPred)

fig = plt.figure(figsize=(8, 5))
axs = fig.subplots(3, 1, sharex=True)
axs[0].plot(test_t[si+test_lead_steps:ei], test_u[si+test_lead_steps:ei, 0], color="blue", linewidth=1, label="True signal")
axs[0].plot(test_t[si+test_lead_steps:ei], test_u_leadPred[:, 0], color="red", label="Lead forecast")
axs[0].set_ylabel(r"$T_W$", fontsize=13)
axs[1].plot(test_t[si+test_lead_steps:ei], test_u[si+test_lead_steps:ei, 1], color="blue", linewidth=1)
axs[1].plot(test_t[si+test_lead_steps:ei], test_u_leadPred[:, 1], color="red")
axs[1].set_ylabel(r"$T_C$", fontsize=13)
axs[1].set_ylim([-2.1, 2.1])
axs[1].set_yticks([-2, 0, 2])
axs[2].plot(test_t[si+test_lead_steps:ei], test_u[si+test_lead_steps:ei, 2], color="blue", linewidth=1)
axs[2].plot(test_t[si+test_lead_steps:ei], test_u_leadPred[:, 2], color="red")
axs[2].set_ylabel(r"$T_E$", fontsize=13)
axs[2].set_xlabel(r"Year", fontsize=13)
for ax in axs.flatten():
    ax.tick_params(labelsize=12, length=4, width=0.8, direction="in")
    ax.set_xlim([test_t[si], test_t[ei]])
    for spine in ax.spines.values():
        spine.set_linewidth(1)
lege = fig.legend(fontsize=13, loc="upper center", ncol=3, fancybox=False, edgecolor="black", bbox_to_anchor=(0.53, 0.93))
fig.tight_layout()
fig.suptitle(r"CGKN for state prediction (6-months lead forecast) (zoomed)", fontsize=14)
fig.subplots_adjust(top=0.83)


# CGKN for Data Assimilation
with torch.no_grad():
    test_mu_z_pred, test_R_z_pred = CGFilter(cgkn,
                                             sigma_hat,
                                             test_u_normalized[:, indices_u1].unsqueeze(-1),
                                             torch.zeros(dim_z, 1),
                                             0.01 * torch.eye(dim_z),
                                             dt)
    test_mu_normalized_pred = cgkn.autoencoder.decoder(test_mu_z_pred.squeeze(-1))
test_mu_pred = (normalizer.std[dim_u1:] + normalizer.eps)*test_mu_normalized_pred + normalizer.mean[dim_u1:]

nnF.mse_loss(test_u[cut_point:, indices_u2], test_mu_pred[cut_point:])

# np.save(path_abs + r"/test_posterior_mean_HW_HC_HE.npy", test_mu_pred)

# UncertaintyNet for Uncertainty Quantification
with torch.no_grad():
    test_mu_std_pred = uncertainty_net(test_u[:, indices_u1])


fig = plt.figure(figsize=(12, 4.5))
axs = fig.subplots(2, 3)
si = int(1500/dt)
ei = int(1540/dt)
axs[0,0].plot(test_t[si:ei], test_u[si:ei, 4], linewidth=1, label="True signal", color="blue")
axs[0,0].plot(test_t[si:ei], test_mu_pred[si:ei, 0], linewidth=1, label="Posterior mean", color="red")
axs[0,0].fill_between(test_t[si:ei], test_mu_pred[si:ei, 0]-2*test_mu_std_pred[si:ei, 0], test_mu_pred[si:ei, 0]+2*test_mu_std_pred[si:ei, 0], color='grey', alpha=0.8, label=r"Uncertainty")
axs[0,0].set_ylabel(r"$H_W$", fontsize=13, labelpad=-10)
axs[0,0].set_xlim([test_t[si], test_t[ei]])
axs[0,1].plot(test_t[si:ei], test_u[si:ei, 5], linewidth=1, color="blue")
axs[0,1].plot(test_t[si:ei], test_mu_pred[si:ei, 1], linewidth=1, color="red")
axs[0,1].fill_between(test_t[si:ei], test_mu_pred[si:ei, 1]-2*test_mu_std_pred[si:ei, 1], test_mu_pred[si:ei, 1]+2*test_mu_std_pred[si:ei, 1], color='grey', alpha=0.8)
axs[0,1].set_ylabel(r"$H_C$", fontsize=13, labelpad=-10)
axs[0,1].set_xlim([test_t[si], test_t[ei]])
axs[0,2].plot(test_t[si:ei], test_u[si:ei, 6], linewidth=1, color="blue")
axs[0,2].plot(test_t[si:ei], test_mu_pred[si:ei, 2], linewidth=1, color="red")
axs[0,2].fill_between(test_t[si:ei], test_mu_pred[si:ei, 2]-2*test_mu_std_pred[si:ei, 2], test_mu_pred[si:ei, 2]+2*test_mu_std_pred[si:ei, 2], color='grey', alpha=0.8)
axs[0,2].set_ylabel(r"$H_E$", fontsize=13, labelpad=-10)
axs[0,2].set_xlim([test_t[si], test_t[ei]])
si = int(1518/dt)
ei = int(1522/dt)
axs[1,0].plot(test_t[si:ei], test_u[si:ei, 4], linewidth=1, color="blue")
axs[1,0].plot(test_t[si:ei], test_mu_pred[si:ei, 0], linewidth=1,  color="red")
axs[1,0].fill_between(test_t[si:ei], test_mu_pred[si:ei, 0]-2*test_mu_std_pred[si:ei, 0], test_mu_pred[si:ei, 0]+2*test_mu_std_pred[si:ei, 0], color='grey', alpha=0.8)
axs[1,0].set_xlabel(r"Year", fontsize=13, rotation=0)
axs[1,0].set_ylabel(r"$H_W$", fontsize=13, labelpad=-10)
axs[1,0].set_xlim([test_t[si], test_t[ei]])
axs[1,0].set_yticks([-20, 0, 20])
axs[1,1].plot(test_t[si:ei], test_u[si:ei, 5], linewidth=1, color="blue")
axs[1,1].plot(test_t[si:ei], test_mu_pred[si:ei, 1], linewidth=1, color="red")
axs[1,1].fill_between(test_t[si:ei], test_mu_pred[si:ei, 1]-2*test_mu_std_pred[si:ei, 1], test_mu_pred[si:ei, 1]+2*test_mu_std_pred[si:ei, 1], color='grey', alpha=0.8)
axs[1,1].set_xlabel(r"Year", fontsize=13, rotation=0)
axs[1,1].set_ylabel(r"$H_C$", fontsize=13, labelpad=-10)
axs[1,1].set_xlim([test_t[si], test_t[ei]])
axs[1,2].plot(test_t[si:ei], test_u[si:ei, 6], linewidth=1, color="blue")
axs[1,2].plot(test_t[si:ei], test_mu_pred[si:ei, 2], linewidth=1, color="red")
axs[1,2].fill_between(test_t[si:ei], test_mu_pred[si:ei, 2]-2*test_mu_std_pred[si:ei, 2], test_mu_pred[si:ei, 2]+2*test_mu_std_pred[si:ei, 2], color='grey', alpha=0.8)
axs[1,2].set_xlabel(r"Year", fontsize=13, rotation=0)
axs[1,2].set_ylabel(r"$H_E$", fontsize=13, labelpad=-10)
axs[1,2].set_xlim([test_t[si], test_t[ei]])
axs[1,2].set_ylim([-25, axs[1,2].get_ylim()[1]])
axs[1,2].set_yticks([-20, 0, 20])
for ax in axs.flatten():
    ax.tick_params(labelsize=13, length=4, width=1, direction="in")
    for spine in ax.spines.values():
        spine.set_linewidth(1)
lege = fig.legend(fontsize=13, loc="upper center", ncol=3, fancybox=False, edgecolor="black", bbox_to_anchor=(0.53, 0.93))
fig.suptitle(r"CGKN for Data Assimilation", fontsize=15)
fig.tight_layout()
fig.subplots_adjust(top=0.8, hspace=0.5)



# CGKN: Number of Parameters
len( torch.nn.utils.parameters_to_vector( cgkn.cgn.parameters() ) )
len( torch.nn.utils.parameters_to_vector( cgkn.autoencoder.encoder.parameters() ) )
len( torch.nn.utils.parameters_to_vector( cgkn.autoencoder.decoder.parameters() ) )

