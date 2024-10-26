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


test_u = test_u.numpy()
test_t = test_t.numpy()


###################################################################
############### Visualization for Lead time Forecast ##############
###################################################################

cgkn_leadPred = np.load(path_abs + r"/Data/L96(40)_CGKN_dimzunit6_test_u_leadPred0u2(400uTo450u).npy")
KoopLinear_leadPred = np.load(path_abs + r"/Data/L96(40)_KoopmanLinear_dimzunit5_test_u_leadPred0u2(400uTo450u).npy")
reg_leadPred = np.load(path_abs + r"/Data/L96(40)_Reg_test_u_leadPred0u2(400uTo450u).npy")


si = int(100/dt) # 400u
ei = int(150/dt) # 450u
test_lead_steps = int(0.2/dt)
vmax = max(np.max(test_u[si+test_lead_steps:ei]), np.max(cgkn_leadPred))
vmin = min(np.min(test_u[si+test_lead_steps:ei]), np.min(cgkn_leadPred))

# Figure 1
fig = plt.figure(figsize=(20, 10))
axs = fig.subplots(2, 1)
c = axs[0].pcolor(test_t[si+test_lead_steps:ei][::10], np.arange(1, I+1), test_u[si+test_lead_steps:ei][::10].T, cmap="jet", vmax=vmax, vmin=vmin)
axs[0].set_title(r"\textbf{(a) True simulation}", fontsize=40)
axs[1].pcolor(test_t[si+test_lead_steps:ei][::10], np.arange(1, I+1), cgkn_leadPred[::10].T, cmap="jet", vmax=vmax, vmin=vmin)
axs[1].set_title(r"\textbf{(b) CGKN}", fontsize=40)
for ax in axs:
    ax.tick_params(labelsize=35, length=15, width=1)
    ax.set_ylabel(r"$i$", fontsize=40, rotation=0)
    ax.set_yticks(np.array([1, 20, 40]))
    ax.set_xticks(np.arange(400, 450+10, 10))
    for spine in ax.spines.values():
        spine.set_linewidth(0)
fig.suptitle(r"\textbf{State forecast at a lead time of 0.2 time units}", fontsize=45)
fig.tight_layout()
fig.subplots_adjust(top=0.84)
fig.savefig(path_abs + "/Figs/L96(40)_LeadForecast.pdf")
fig.savefig(path_abs + "/Figs/L96(40)_LeadForecast.png")

# Figure 2
fig = plt.figure(figsize=(20, 10))
axs = fig.subplots(2, 1)
axs[0].plot(test_t[si+test_lead_steps:ei], test_u[si+test_lead_steps:ei, 0], color="black", linewidth=4, label=r"\textbf{True signal}")
axs[0].plot(test_t[si+test_lead_steps:ei], KoopLinear_leadPred[:, 0], color="blue", linewidth=3, label=r"\textbf{Simplified KoopNet}")
axs[0].plot(test_t[si+test_lead_steps:ei], cgkn_leadPred[:, 0], color="red", linewidth=3.5, label=r"\textbf{CGKN}")
axs[0].plot(test_t[si+test_lead_steps:ei], reg_leadPred[:, 0], color="green", linewidth=3, label=r"\textbf{CG-Reg}")
axs[0].set_ylabel(r"$x_1$", fontsize=40, rotation=0)
axs[0].set_xlim([test_t[si], test_t[ei]])
axs[1].plot(test_t[si+test_lead_steps:ei], test_u[si+test_lead_steps:ei, 1], color="black", linewidth=4)
axs[1].plot(test_t[si+test_lead_steps:ei], KoopLinear_leadPred[:, 1], color="blue", linewidth=3)
axs[1].plot(test_t[si+test_lead_steps:ei], cgkn_leadPred[:, 1], color="red", linewidth=3.5)
axs[1].plot(test_t[si+test_lead_steps:ei], reg_leadPred[:, 1], color="green", linewidth=3)
axs[1].set_xlabel(r"$t$", fontsize=40)
axs[1].set_ylabel(r"$x_2$", fontsize=40, rotation=0)
axs[1].set_xlim([test_t[si], test_t[ei]])
lege_handles, lege_labels = axs[0].get_legend_handles_labels()
lege = fig.legend(handles=[lege_handles[i] for i in [0,2,1,3]] , labels=[lege_labels[i] for i in [0,2,1,3]] , fontsize=35, loc="upper center", ncol=4, fancybox=False, edgecolor="black", bbox_to_anchor=(0.5, 1))
lege.get_frame().set_linewidth(1)
for ax in axs.flatten():
    ax.tick_params(labelsize=35, length=20, width=1, direction="in", top=True, bottom=True, left=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
fig.tight_layout()
fig.subplots_adjust(top=0.88)
fig.savefig(path_abs + "/Figs/L96(40)_LeadForecast2.pdf")
fig.savefig(path_abs + "/Figs/L96(40)_LeadForecast2.png")



##################################################################
############### Visualization for Data Assimilation ##############
##################################################################

trueModel_test_mu = np.load(path_abs + r"/Data/L96(40)_TrueModel_test_mu.npy")
trueModel_test_mu_std = np.load(path_abs + r"/Data/L96(40)_TrueModel_test_mu_std.npy")
cgkn_test_mu_pred = np.load(path_abs + r"/Data/L96(40)_CGKN_dimzunit6_test_mu_pred.npy")
cgkn_test_mu_std_pred = np.load(path_abs + r"/Data/L96(40)_CGKN_dimzunit6_test_mu_std_pred.npy")
KoopLinear_test_mu_pred = np.load(path_abs + r"/Data/L96(40)_KoopmanLinear_dimzunit5_test_mu_pred.npy")
KoopLinear_test_mu_std_pred = np.load(path_abs + r"/Data/L96(40)_KoopmanLinear_dimzunit5_test_mu_std_pred.npy")
reg_test_mu_pred = np.load(path_abs + r"/Data/L96(40)_Reg_test_mu_pred.npy")
reg_test_mu_std_pred = np.load(path_abs + r"/Data/L96(40)_Reg_test_mu_std_pred.npy")

# Figure DA
fig = plt.figure(figsize=(10, 9))
subfigs = fig.subfigures(2, 2, hspace=-0.15, wspace=-0.1)
for subfig in subfigs.flatten():
    subfig.subplots_adjust(hspace=0.4)
axs0 = subfigs[0,0].subplots(2, 1)
axs1 = subfigs[0,1].subplots(2, 1)
axs2 = subfigs[1,0].subplots(2, 1)
axs3 = subfigs[1,1].subplots(2, 1)
si = 10000  # 400s
ei = 15000  # 450s
axs0[0].plot(test_t[si:ei], test_u[si:ei, 1], linewidth=2, label=r"\textbf{True signal}", color="black")
axs0[0].plot(test_t[si:ei], trueModel_test_mu[si:ei, 0], linewidth=1, label=r"\textbf{True model with EnKBF}", color="orange")
axs0[0].fill_between(test_t[si:ei], trueModel_test_mu[si:ei, 0]-2*trueModel_test_mu_std[si:ei, 0],  trueModel_test_mu[si:ei, 0]+2*trueModel_test_mu_std[si:ei, 0], color='grey', alpha=0.5, label=r"\textbf{Uncertainty}")
axs0[0].set_ylabel(r"$x_2$", fontsize=15, rotation=0, labelpad=-7)
axs0[0].set_title(r"\textbf{(a) True model}", fontsize=15, rotation=0)
axs1[0].plot(test_t[si:ei], test_u[si:ei, 1], linewidth=2, color="black")
axs1[0].plot(test_t[si:ei], cgkn_test_mu_pred[si:ei, 0], linewidth=1, color="red", label=r"\textbf{CGKN}")
axs1[0].fill_between(test_t[si:ei], cgkn_test_mu_pred[si:ei, 0]-2*cgkn_test_mu_std_pred[si:ei, 0],  cgkn_test_mu_pred[si:ei, 0]+2*cgkn_test_mu_std_pred[si:ei, 0], color='grey', alpha=0.5)
axs1[0].set_ylabel(r"$x_2$", fontsize=15, rotation=0, labelpad=-7)
axs1[0].set_title(r"\textbf{(b) CGKN}", fontsize=15, rotation=0)
axs2[0].plot(test_t[si:ei], test_u[si:ei, 1], linewidth=2, color="black")
axs2[0].plot(test_t[si:ei], KoopLinear_test_mu_pred[si:ei, 0], linewidth=1, color="blue", label=r"\textbf{Simplified KoopNet}")
axs2[0].fill_between(test_t[si:ei], KoopLinear_test_mu_pred[si:ei, 0]-2*KoopLinear_test_mu_std_pred[si:ei, 0],  KoopLinear_test_mu_pred[si:ei, 0]+2*KoopLinear_test_mu_std_pred[si:ei, 0], color='grey', alpha=0.5)
axs2[0].set_ylabel(r"$x_2$", fontsize=15, rotation=0, labelpad=-7)
axs2[0].set_title(r"\textbf{(c) Simplified KoopNet}", fontsize=15, rotation=0)
axs3[0].plot(test_t[si:ei], test_u[si:ei, 1], linewidth=2, color="black")
axs3[0].plot(test_t[si:ei], reg_test_mu_pred[si:ei, 0], linewidth=1, color="green", label=r"\textbf{CG-Reg}")
axs3[0].fill_between(test_t[si:ei], reg_test_mu_pred[si:ei, 0]-2*reg_test_mu_std_pred[si:ei, 0],  reg_test_mu_pred[si:ei, 0]+2*reg_test_mu_std_pred[si:ei, 0], color='grey', alpha=0.5)
axs3[0].set_ylabel(r"$x_2$", fontsize=15, rotation=0, labelpad=-7)
axs3[0].set_title(r"\textbf{(d) CG-Reg}", fontsize=15, rotation=0)
for sf in subfigs.flatten():
    sf.get_axes()[0].set_xlim([test_t[si], test_t[ei]])
si = 12000  # 420s
ei = 13000  # 430s
axs0[1].plot(test_t[si:ei], test_u[si:ei, 1], linewidth=2,  color="black")
axs0[1].plot(test_t[si:ei], trueModel_test_mu[si:ei, 0], linewidth=1, color="orange")
axs0[1].fill_between(test_t[si:ei], trueModel_test_mu[si:ei, 0]-2*trueModel_test_mu_std[si:ei, 0],  trueModel_test_mu[si:ei, 0]+2*trueModel_test_mu_std[si:ei, 0], color='grey', alpha=0.5)
axs0[1].set_xlabel(r"$t$", fontsize=15)
axs1[1].plot(test_t[si:ei], test_u[si:ei, 1], linewidth=2, color="black")
axs1[1].plot(test_t[si:ei], cgkn_test_mu_pred[si:ei, 0], linewidth=1, color="red")
axs1[1].fill_between(test_t[si:ei], cgkn_test_mu_pred[si:ei, 0]-2*cgkn_test_mu_std_pred[si:ei, 0],  cgkn_test_mu_pred[si:ei, 0]+2*cgkn_test_mu_std_pred[si:ei, 0], color='grey', alpha=0.5)
axs1[1].set_xlabel(r"$t$", fontsize=15)
axs2[1].plot(test_t[si:ei], test_u[si:ei, 1], linewidth=2, color="black")
axs2[1].plot(test_t[si:ei], KoopLinear_test_mu_pred[si:ei, 0], linewidth=1, color="blue")
axs2[1].fill_between(test_t[si:ei], KoopLinear_test_mu_pred[si:ei, 0]-2*KoopLinear_test_mu_std_pred[si:ei, 0],  KoopLinear_test_mu_pred[si:ei, 0]+2*KoopLinear_test_mu_std_pred[si:ei, 0], color='grey', alpha=0.5)
axs2[1].set_xlabel(r"$t$", fontsize=15)
axs3[1].plot(test_t[si:ei], test_u[si:ei, 1], linewidth=2, color="black")
axs3[1].plot(test_t[si:ei], reg_test_mu_pred[si:ei, 0], linewidth=1, color="green")
axs3[1].fill_between(test_t[si:ei], reg_test_mu_pred[si:ei, 0]-2*reg_test_mu_std_pred[si:ei, 0],  reg_test_mu_pred[si:ei, 0]+2*reg_test_mu_std_pred[si:ei, 0], color='grey', alpha=0.5)
axs3[1].set_xlabel(r"$t$", fontsize=15)
for sf in subfigs.flatten():
    sf.get_axes()[0].set_ylim([-12, 15])
    # sf.get_axes()[0].set_yticks([-2, -1, 0, 1])
    sf.get_axes()[1].set_ylim([-6, 11])
    sf.get_axes()[1].set_yticks([-5, 0, 5, 10])
    sf.get_axes()[1].set_xlim([test_t[si], test_t[ei]])
    for ax in sf.get_axes():
        ax.tick_params(labelsize=14, length=8, width=1, direction="in", top=True, bottom=True, left=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(1)
lege_handles, lege_labels = axs0[0].get_legend_handles_labels()
lege_handles.extend([axs1[0].get_legend_handles_labels()[0][0], axs2[0].get_legend_handles_labels()[0][0], axs3[0].get_legend_handles_labels()[0][0]])
lege_labels.extend([axs1[0].get_legend_handles_labels()[1][0], axs2[0].get_legend_handles_labels()[1][0], axs3[0].get_legend_handles_labels()[1][0]])
lege = subfigs[0,0].legend(handles=[lege_handles[i] for i in [0, 2, 1, 4, 3, 5]], labels=[lege_labels[i] for i in [0, 2, 1, 4, 3, 5]], fontsize=15, loc="upper center", ncol=3, fancybox=False, edgecolor="black", bbox_to_anchor=(0.95, 1)) #
lege.get_frame().set_linewidth(0.8)
fig.subplots_adjust(top=0.75)
fig.savefig(path_abs + "/Figs/L96(40)_DA.pdf")
fig.savefig(path_abs + "/Figs/L96(40)_DA.png")



