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
################# Data Generation #################
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


test_u = test_u.numpy()
test_t = test_t.numpy()


##################################################################
############### Visualization for Data Assimilation ##############
##################################################################
trueModel_test_mu = np.load(path_abs + r"/Data/PSBSE_TrueModel_test_mu.npy")
trueModel_test_mu_std = np.load(path_abs + r"/Data/PSBSE_TrueModel_test_mu_std.npy")
cgkn_test_mu_pred = np.load(path_abs + r"/Data/PSBSE_CGKN_test_mu_pred.npy")
cgkn_test_mu_std_pred = np.load(path_abs + r"/Data/PSBSE_CGKN_test_mu_std_pred.npy")
KoopLinear_test_mu_pred = np.load(path_abs + r"/Data/PSBSE_KoopLinear_dimz10_test_mu_pred.npy")
KoopLinear_test_mu_std_pred = np.load(path_abs + r"/Data/PSBSE_KoopLinear_dimz10_test_mu_std_pred.npy")
reg_test_mu_pred = np.load(path_abs + r"/Data/PSBSE_Reg_test_mu_pred.npy")
reg_test_mu_std_pred = np.load(path_abs + r"/Data/PSBSE_Reg_test_mu_std_pred.npy")


si = 5000 # 15000  # 950s
ei = 7000 # 17000  # 970s
test_t_part = test_t[si:ei]
test_u_part = test_u[si:ei]
trueModel_test_mu = trueModel_test_mu[si:ei]
trueModel_test_mu_std = trueModel_test_mu_std[si:ei]
cgkn_test_mu_pred = cgkn_test_mu_pred[si:ei]
cgkn_test_mu_std_pred = cgkn_test_mu_std_pred[si:ei]
KoopLinear_test_mu_pred = KoopLinear_test_mu_pred[si:ei]
KoopLinear_test_mu_std_pred = KoopLinear_test_mu_std_pred[si:ei]
reg_test_mu_pred = reg_test_mu_pred[si:ei]
reg_test_mu_std_pred = reg_test_mu_std_pred[si:ei]



fig = plt.figure(figsize=(10.5, 8.5))
subfigs = fig.subfigures(2, 2, hspace=-0.15, wspace=-0.1)
# True Model
axs0 = subfigs[0,0].subplots(2, 1, sharex=True)
axs0[0].plot(test_t_part, test_u_part[:, 1], linewidth=1.5, label=r"\textbf{True signal}", color="black")
axs0[0].plot(test_t_part, trueModel_test_mu[:, 0], linewidth=1, label=r"\textbf{True Model with EnKBF}", color="orange")
axs0[0].fill_between(test_t_part, trueModel_test_mu[:, 0]-2*trueModel_test_mu_std[:, 0],  trueModel_test_mu[:, 0]+2*trueModel_test_mu_std[:, 0], color='grey', alpha=0.5, label=r"\textbf{Uncertainty}")
axs0[0].set_ylabel(r"$y$", fontsize=15, rotation=0)
axs0[0].set_title(r"\textbf{(a) True model}", fontsize=15, rotation=0)
# axs0[0].tick_params(labelsize=14, length=3, width=1, direction="in")
axs0[1].plot(test_t_part, test_u_part[:, 2], linewidth=1.5, color="black")
axs0[1].plot(test_t_part, trueModel_test_mu[:, 1], linewidth=1, color="orange")
axs0[1].fill_between(test_t_part, trueModel_test_mu[:, 1]-2*trueModel_test_mu_std[:,1], trueModel_test_mu[:, 1]+2*trueModel_test_mu_std[:,1], color='grey', alpha=0.5)
axs0[1].set_ylabel(r"$z$", fontsize=15, rotation=0)
axs0[1].set_xlabel(r"$t$", fontsize=15)
# CGKN
axs1 = subfigs[0,1].subplots(2, 1, sharex=True)
axs1[0].plot(test_t_part, test_u_part[:,1], linewidth=1.5, color="black")
axs1[0].plot(test_t_part, cgkn_test_mu_pred[:,0], linewidth=1, color="red", label=r"\textbf{CGKN}")
axs1[0].fill_between(test_t_part, cgkn_test_mu_pred[:,0]-2*cgkn_test_mu_std_pred[:,0], cgkn_test_mu_pred[:,0]+2*cgkn_test_mu_std_pred[:,0], color='grey', alpha=0.5)
axs1[0].set_ylabel(r"$y$", fontsize=15, rotation=0)
axs1[0].set_title(r"\textbf{(b) CGKN}", fontsize=15, rotation=0)
axs1[1].plot(test_t_part, test_u_part[:, 2], linewidth=1.5, color="black")
axs1[1].plot(test_t_part, cgkn_test_mu_pred[:, 1], linewidth=1, color="red")
axs1[1].fill_between(test_t_part, cgkn_test_mu_pred[:, 1]-2*cgkn_test_mu_std_pred[:, 1], cgkn_test_mu_pred[:, 1]+2*cgkn_test_mu_std_pred[:, 1], color='grey', alpha=0.5)
axs1[1].set_ylabel(r"$z$", fontsize=15, rotation=0)
axs1[1].set_xlabel(r"$t$", fontsize=15)
# KoopLinear
axs2 = subfigs[1,0].subplots(2, 1, sharex=True)
axs2[0].plot(test_t_part, test_u_part[:, 1], linewidth=1.5, color="black")
axs2[0].plot(test_t_part, KoopLinear_test_mu_pred[:, 0], linewidth=1, color="blue", label=r"\textbf{Simplified KoopNet}")
axs2[0].fill_between(test_t_part, KoopLinear_test_mu_pred[:,0]-2*KoopLinear_test_mu_std_pred[:,0],KoopLinear_test_mu_pred[:,0]+2*KoopLinear_test_mu_std_pred[:,0], color='grey', alpha=0.5)
axs2[0].set_ylabel(r"$y$", fontsize=15, rotation=0)
axs2[0].set_title(r"\textbf{(c) Simplified KoopNet}", fontsize=15, rotation=0)
axs2[1].plot(test_t_part, test_u_part[:, 2], linewidth=1.5, color="black")
axs2[1].plot(test_t_part, KoopLinear_test_mu_pred[:, 1], linewidth=1, color="blue")
axs2[1].fill_between(test_t_part, KoopLinear_test_mu_pred[:, 1]-2*KoopLinear_test_mu_std_pred[:, 1], KoopLinear_test_mu_pred[:, 1]+2*KoopLinear_test_mu_std_pred[:, 1], color='grey', alpha=0.5)
axs2[1].set_ylabel(r"$z$", fontsize=15, rotation=0)
axs2[1].set_xlabel(r"$t$", fontsize=15)
# Reg
axs3 = subfigs[1,1].subplots(2, 1, sharex=True)
axs3[0].plot(test_t_part, test_u_part[:, 1], linewidth=1.5, color="black")
axs3[0].plot(test_t_part, reg_test_mu_pred[:, 0], linewidth=1, color="green", label=r"\textbf{CG-Reg}")
axs3[0].fill_between(test_t_part, reg_test_mu_pred[:, 0]-2*reg_test_mu_std_pred[:, 0], reg_test_mu_pred[:, 0]+2*reg_test_mu_std_pred[:, 0], color='grey', alpha=0.5)
axs3[0].set_ylabel(r"$y$", fontsize=15, rotation=0)
axs3[0].set_title(r"\textbf{(d) CG-Reg}", fontsize=15, rotation=0)
axs3[1].plot(test_t_part, test_u_part[:, 2], linewidth=1.5, color="black")
axs3[1].plot(test_t_part, reg_test_mu_pred[:, 1], linewidth=1, color="green")
axs3[1].fill_between(test_t_part, reg_test_mu_pred[:, 1]-2*reg_test_mu_std_pred[:, 1], reg_test_mu_pred[:, 1]+2*reg_test_mu_std_pred[:, 1], color='grey', alpha=0.5)
axs3[1].set_ylabel(r"$z$", fontsize=15, rotation=0)
axs3[1].set_xlabel(r"$t$", fontsize=15)
for sf in subfigs.flatten():
    sf.get_axes()[0].set_ylim([-2.5, 1.8])
    sf.get_axes()[0].set_yticks([-2, -1, 0, 1])
    sf.get_axes()[1].set_ylim([-2.2, 1.5])
    sf.get_axes()[1].set_yticks([-2, -1, 0, 1])
    for ax in sf.get_axes():
        ax.set_xlim([test_t_part[0], test_t_part[-1]])
        ax.set_xticks([850, 855, 860, 865, 870])
        ax.tick_params(labelsize=14, length=6, width=0.8, direction="in", top=True, bottom=True, left=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(1)
lege_handles, lege_labels = axs0[0].get_legend_handles_labels()
lege_handles.extend([axs1[0].get_legend_handles_labels()[0][0], axs2[0].get_legend_handles_labels()[0][0], axs3[0].get_legend_handles_labels()[0][0]])
lege_labels.extend([axs1[0].get_legend_handles_labels()[1][0], axs2[0].get_legend_handles_labels()[1][0], axs3[0].get_legend_handles_labels()[1][0]])
lege = subfigs[0,0].legend(handles=[lege_handles[i] for i in [0, 2, 1, 4, 3, 5]], labels=[lege_labels[i] for i in [0, 2, 1, 4, 3, 5]], fontsize=15, loc="upper center", ncol=3, fancybox=False, edgecolor="black", bbox_to_anchor=(0.95, 1)) #
lege.get_frame().set_linewidth(0.8)
fig.subplots_adjust(top=0.75)
fig.savefig(path_abs + "/Figs/PSBSE_DA.pdf")
fig.savefig(path_abs + "/Figs/PSBSE_DA.png")

