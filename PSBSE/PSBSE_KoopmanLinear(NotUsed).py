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

class AutoEncoder1(nn.Module):
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

class AutoEncoder2(nn.Module):
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

# Fully linear with z1 and z2
class CGN(nn.Module):
    def __init__(self, dim_z1, dim_z2):
        super().__init__()
        self.input_size = dim_z1
        self.f1_size = (dim_z1, 1)
        self.g1_size = (dim_z1, dim_z2)
        self.f2_size = (dim_z2, 1)
        self.g2_size = (dim_z2, dim_z2)
        self.output_size = np.prod(self.f1_size) + np.prod(self.g1_size) + np.prod(self.f2_size) + np.prod(self.g2_size)
        self.net = nn.Linear(self.input_size, self.output_size)

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
    def __init__(self, autoencoder1, autoencoder2, cgn):
        super().__init__()
        self.autoencoder1 = autoencoder1
        self.autoencoder2 = autoencoder2
        self.cgn = cgn

    def forward(self, t, u_extended):
        # Matrix Form Computation
        dim_z1 = self.cgn.input_size
        z1 = u_extended[:, :dim_z1]
        z2 = u_extended[:, dim_z1:]
        f1, g1, f2, g2 = self.cgn(z1)
        z2 = z2.unsqueeze(-1)
        z1_dot = f1 + g1@z2
        z2_dot = f2 + g2@z2
        return torch.cat([z1_dot.squeeze(-1), z2_dot.squeeze(-1)], dim=-1)


########################################################
################# Train cgkn (Stage1)  #################
########################################################
dim_u1 = 1
dim_u2 = 2
dim_z1 = 12
dim_z2 = 12
dim_f1 = (dim_z1, 1)
dim_g1 = (dim_z1, dim_z2)
dim_f2 = (dim_z2, 1)
dim_g2 = (dim_z2, dim_z2)


# Stage1: Train cgkn with loss_forecast + loss_ae + loss_forecast_z1 + loss_forecast_z2
epochs = 500
batch_size = 200
train_tensor = torch.utils.data.TensorDataset(train_u, train_u_dot)
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=batch_size)
train_num_batches = len(train_loader)
iterations = epochs * train_num_batches
train_loss_forecast_history = []
train_loss_ae1_history = []
train_loss_ae2_history = []
train_loss_forecast_z1_history = []
train_loss_forecast_z2_history = []

autoencoder1 = AutoEncoder1(dim_u1, dim_z1).to(device)
autoencoder2 = AutoEncoder2(dim_u2, dim_z2).to(device)
cgn = CGN(dim_z1, dim_z2).to(device)
cgkn = CGKN(autoencoder1, autoencoder2, cgn).to(device)
optimizer = torch.optim.Adam(cgkn.parameters(), lr=1e-3)
for ep in range(1, epochs+1):
    start_time = time.time()

    train_loss_forecast = 0.
    train_loss_ae1 = 0.
    train_loss_ae2 = 0.
    train_loss_forecast_z1 = 0.
    train_loss_forecast_z2 = 0.
    for u, u_dot in train_loader:
        u, u_dot = u.to(device), u_dot.to(device)
        optimizer.zero_grad()

        # Encoder
        z1 = cgkn.autoencoder1.encoder(u[:, indices_u1])
        z2 = cgkn.autoencoder2.encoder(u[:, indices_u2])
        # Decoder
        u1_ae = cgkn.autoencoder1.decoder(z1)
        u2_ae = cgkn.autoencoder2.decoder(z2)
        loss_ae1 = nnF.mse_loss(u[:, indices_u1], u1_ae)
        loss_ae2 = nnF.mse_loss(u[:, indices_u2], u2_ae)

        u_extended = torch.cat([z1, z2], dim=-1)
        # Latent One-Step Future State
        u_extended_dot_pred = cgkn(None, u_extended)
        u_extended_pred = u_extended + u_extended_dot_pred*dt
        # Decoder
        u1_pred = cgkn.autoencoder1.decoder(u_extended_pred[:, :dim_z1])
        u2_pred = cgkn.autoencoder2.decoder(u_extended_pred[:, dim_z1:])
        u_pred = torch.cat([u1_pred, u2_pred], dim=-1)
        loss_forecast = nnF.mse_loss(u+u_dot*dt, u_pred)

        # Encoder (Future State)
        z1 = cgkn.autoencoder1.encoder(u[:, indices_u1]+u_dot[:, indices_u1]*dt)
        z2 = cgkn.autoencoder2.encoder(u[:, indices_u2]+u_dot[:, indices_u2]*dt)
        loss_forecast_z1 = nnF.mse_loss(z1, u_extended_pred[:, :dim_z1])
        loss_forecast_z2 = nnF.mse_loss(z2, u_extended_pred[:, dim_z1:])

        loss_total = loss_forecast + loss_ae1 + loss_forecast_z1 + loss_ae2 + loss_forecast_z2
        loss_total.backward()
        optimizer.step()

        train_loss_forecast += loss_forecast.item()
        train_loss_ae1 += loss_ae1.item()
        train_loss_ae2 += loss_ae2.item()
        train_loss_forecast_z1 += loss_forecast_z1.item()
        train_loss_forecast_z2 += loss_forecast_z2.item()
    train_loss_forecast /= train_num_batches
    train_loss_ae1 /= train_num_batches
    train_loss_ae2 /= train_num_batches
    train_loss_forecast_z1 /= train_num_batches
    train_loss_forecast_z2 /= train_num_batches

    train_loss_forecast_history.append(train_loss_forecast)
    train_loss_ae1_history.append(train_loss_ae1)
    train_loss_ae2_history.append(train_loss_ae2)
    train_loss_forecast_z1_history.append(train_loss_forecast_z1)
    train_loss_forecast_z2_history.append(train_loss_forecast_z2)

    end_time = time.time()
    print(ep, "time:", round(end_time - start_time, 4),
          " loss fore:", round(train_loss_forecast, 4),
          " loss ae1:", round(train_loss_ae1, 4),
          " loss ae2:", round(train_loss_ae2, 4),
          " loss fore z1:", round(train_loss_forecast_z1, 4),
          " loss fore z2:", round(train_loss_forecast_z2, 4))


# # Model Diagnosis in Physical Space
# cgkn.to("cpu")
# with torch.no_grad():
#     train_z1 = cgkn.autoencoder1.encoder(train_u[:, indices_u1])
#     train_z2 = cgkn.autoencoder2.encoder(train_u[:, indices_u2])
# train_u_extended = torch.cat([train_z1, train_z2], dim=-1)
# with torch.no_grad():
#     train_u_extended_dot_pred = cgkn(None, train_u_extended)
# train_u_extended_pred = train_u_extended + train_u_extended_dot_pred * dt
# with torch.no_grad():
#     train_u1_pred = cgkn.autoencoder1.decoder(train_u_extended_pred[:, :dim_z1])
#     train_u2_pred = cgkn.autoencoder2.decoder(train_u_extended_pred[:, dim_z1:])
# train_u_pred = torch.cat([train_u1_pred, train_u2_pred], dim=-1)
# train_u_dot_pred = (train_u_pred - train_u)/dt
# nnF.mse_loss(train_u_dot[:, 0], train_u_dot_pred[:, 0])
# nnF.mse_loss(train_u_dot[:, 1], train_u_dot_pred[:, 1])
# nnF.mse_loss(train_u_dot[:, 2], train_u_dot_pred[:, 2])
# cgkn.to(device)


#################################################################
################# Noise Coefficient & CGFilter  #################
#################################################################
# cgkn.to("cpu")
# with torch.no_grad():
#     train_z1 = cgkn.autoencoder1.encoder(train_u[:, indices_u1])
#     train_z2 = cgkn.autoencoder2.encoder(train_u[:, indices_u2])
# train_u_extended = torch.cat([train_z1, train_z2], dim=-1)
# train_u_extended_dot = torch.diff(train_u_extended, dim=0)/dt
# with torch.no_grad():
#     train_u_extended_dot_pred = cgkn(None, train_u_extended)
# sigma_hat = torch.sqrt( dt*torch.mean( (train_u_extended_dot - train_u_extended_dot_pred[:-1])**2, dim=0 ) )
# sigma_hat = 10*sigma_hat
# cgkn.to(device)

sigma_hat = torch.ones(dim_z1+dim_z2)


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

# Stage 2: Train cgkn with loss_forecast + loss_da + loss_ae1 + loss_forecast_z1 + loss_ae2 + loss_forecast_z2
short_steps = int(0.1/dt)
long_steps = int(50/dt)
cut_point = int(3/dt)

epochs = 10000
train_batch_size = 400
train_loss_forecast_history = []
train_loss_da_history = []
train_loss_ae1_history = []
train_loss_ae2_history = []
train_loss_forecast_z1_history = []
train_loss_forecast_z2_history = []
# Re-initialize Model
autoencoder1 = AutoEncoder1(dim_u1, dim_z1).to(device)
autoencoder2 = AutoEncoder2(dim_u2, dim_z2).to(device)
cgn = CGN(dim_z1, dim_z2).to(device)
cgkn = CGKN(autoencoder1,autoencoder2, cgn).to(device)
optimizer = torch.optim.Adam(cgkn.parameters(), lr=1e-3)
for ep in range(1, epochs+1):
    start_time = time.time()
    optimizer.zero_grad()

    head_indices_short = torch.from_numpy(np.random.choice(Ntrain-short_steps+1, size=train_batch_size, replace=False))
    u_short = torch.stack([train_u[idx:idx + short_steps] for idx in head_indices_short], dim=1).to(device)
    t_short = train_t[:short_steps].to(device)

    # Encoder
    z1_short = cgkn.autoencoder1.encoder(u_short[:, :, indices_u1])
    z2_short = cgkn.autoencoder2.encoder(u_short[:, :, indices_u2])
    u_extended0_short = torch.cat([z1_short[0], z2_short[0]], dim=-1)

    # Integration
    u_extended_pred_short = torchdiffeq.odeint(cgkn, u_extended0_short, t_short, method="rk4", options={"step_size":0.001})
    loss_forecast_z1 = nnF.mse_loss(z1_short, u_extended_pred_short[:, :, :dim_z1])
    loss_forecast_z2 = nnF.mse_loss(z2_short, u_extended_pred_short[:, :, dim_z1:])
    # Decoder
    u1_pred_short = cgkn.autoencoder1.decoder(u_extended_pred_short[:, :, :dim_z1])
    u2_pred_short = cgkn.autoencoder2.decoder(u_extended_pred_short[:, :, dim_z1:])
    u_pred_short = torch.cat([u1_pred_short, u2_pred_short], dim=-1)
    loss_forecast = nnF.mse_loss(u_short, u_pred_short)

    head_idx_long = torch.from_numpy( np.random.choice(Ntrain-long_steps+1, size=1) )
    u_long = train_u[head_idx_long:head_idx_long + long_steps].to(device)
    t_long = train_t[head_idx_long:head_idx_long + long_steps].to(device)
    # autoendocer
    z1_long = cgkn.autoencoder1.encoder(u_long[:, indices_u1])
    z2_long = cgkn.autoencoder2.encoder(u_long[:, indices_u2])
    u1_ae_long = cgkn.autoencoder1.decoder(z1_long)
    u2_ae_long = cgkn.autoencoder2.decoder(z2_long)
    loss_ae1 = nnF.mse_loss(u_long[:, indices_u1], u1_ae_long)
    loss_ae2 = nnF.mse_loss(u_long[:, indices_u2], u2_ae_long)

    # Filter
    mu_z2_pred_long = CGFilter(cgkn, sigma_hat.to(device), z1_long.unsqueeze(-1), mu0=torch.zeros(dim_z2, 1).to(device), R0=0.01*torch.eye(dim_z2).to(device), dt=dt)[0].squeeze(-1)
    # Decoder
    mu_pred_long = cgkn.autoencoder2.decoder(mu_z2_pred_long[cut_point:])
    loss_da = nnF.mse_loss(u_long[cut_point:, indices_u2], mu_pred_long)

    loss_total = loss_forecast + loss_da + loss_ae1 + loss_forecast_z1 + loss_ae2 + loss_forecast_z2
    if torch.isnan(loss_total):
        print(ep, "nan")
        continue
    loss_total.backward()
    optimizer.step()
    train_loss_forecast_history.append(loss_forecast.item())
    train_loss_da_history.append(loss_da.item())
    train_loss_ae1_history.append(loss_ae1.item())
    train_loss_forecast_z1_history.append(loss_forecast_z1.item())
    train_loss_ae2_history.append(loss_ae2.item())
    train_loss_forecast_z2_history.append(loss_forecast_z2.item())

    end_time = time.time()
    print(ep, " time:", round(end_time-start_time, 4),
          " loss fore:", round(loss_forecast.item(), 4),
          " loss da:", round(loss_da.item(),4),
          " loss ae1:", round(loss_ae1.item(),4),
          " loss ae2:", round(loss_ae2.item(),4),
          " loss fore z1:", round(loss_forecast_z1.item(), 4),
          " loss fore z2:", round(loss_forecast_z2.item(), 4))

torch.save(cgkn.state_dict(), r"/Users/chuanqichen/File/PythonProject/CGKN/PSBSE/Models/PSBSE_KoopmanLinear_dimz1_12_dimz2_12.pt")
np.save(r"/Users/chuanqichen/File/PythonProject/CGKN/PSBSE/Models/PSBSE_KoopmanLinear_dimz1_12_dimz2_12_train_loss_forecast_history", train_loss_forecast_history)
np.save(r"/Users/chuanqichen/File/PythonProject/CGKN/PSBSE/Models/PSBSE_KoopmanLinear_dimz1_12_dimz2_12_train_loss_da_history", train_loss_da_history)
np.save(r"/Users/chuanqichen/File/PythonProject/CGKN/PSBSE/Models/PSBSE_KoopmanLinear_dimz1_12_dimz2_12_train_loss_ae1_history", train_loss_ae1_history)
np.save(r"/Users/chuanqichen/File/PythonProject/CGKN/PSBSE/Models/PSBSE_KoopmanLinear_dimz1_12_dimz2_12_train_loss_ae2_history", train_loss_ae2_history)
np.save(r"/Users/chuanqichen/File/PythonProject/CGKN/PSBSE/Models/PSBSE_KoopmanLinear_dimz1_12_dimz2_12_train_loss_forecast_z1_history", train_loss_forecast_z1_history)
np.save(r"/Users/chuanqichen/File/PythonProject/CGKN/PSBSE/Models/PSBSE_KoopmanLinear_dimz1_12_dimz2_12_train_loss_forecast_z2_history", train_loss_forecast_z2_history)



#############################################
################# Test cgkn #################
#############################################

cgkn.to("cpu")

# CGKN for State Prediction
test_short_steps = int(0.2/dt)
test_u0 = test_u[::test_short_steps]
with torch.no_grad():
    test_z0 = cgkn.autoencoder.encoder(test_u0[:, indices_u2])
test_u_extended0 = torch.cat([test_u0[:, indices_u1], test_z0], dim=-1)
with torch.no_grad():
    test_u_extended_shortPred = torchdiffeq.odeint(cgkn, test_u_extended0, t[:test_short_steps], method="rk4", options={"step_size":0.001})
    test_u2_shortPred = cgkn.autoencoder.decoder(test_u_extended_shortPred[:, :, dim_u1:])
test_u_shortPred = torch.cat([test_u_extended_shortPred[:, :, :dim_u1], test_u2_shortPred], dim=-1)
test_u_shortPred = test_u_shortPred.permute(1, 0, 2).reshape(Ntest, -1)
nnF.mse_loss(test_u, test_u_shortPred)


# CGKN for Data Assimilation
with torch.no_grad():
    test_mu_z_pred, test_R_z_pred = CGFilter(cgkn,
                                             sigma_hat,
                                             test_u[:, indices_u1].unsqueeze(-1),
                                             torch.zeros(dim_z, 1),
                                             0.01 * torch.eye(dim_z),
                                             dt)
    test_mu_pred = cgkn.autoencoder.decoder(test_mu_z_pred.squeeze(-1))

nnF.mse_loss(test_u[cut_point:, indices_u2], test_mu_pred[cut_point:])



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


# CGKN: Number of Parameters
len( torch.nn.utils.parameters_to_vector( cgkn.cgn.parameters() ) )
len( torch.nn.utils.parameters_to_vector( cgkn.autoencoder1.encoder.parameters() ) )
len( torch.nn.utils.parameters_to_vector( cgkn.autoencoder1.decoder.parameters() ) )
len( torch.nn.utils.parameters_to_vector( cgkn.autoencoder2.encoder.parameters() ) )
len( torch.nn.utils.parameters_to_vector( cgkn.autoencoder2.decoder.parameters() ) )

