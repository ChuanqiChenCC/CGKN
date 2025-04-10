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

device = "cpu"
torch.manual_seed(0)
np.random.seed(0)

plt.rc("text", usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

#########################################
############# Data Import ###############
#########################################

train_ENSO_mat = sp.io.loadmat("Data/ENSO_model_data1.mat")
test_ENSO_mat = sp.io.loadmat("Data/ENSO_model_data2.mat")
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
dim_u1 = len(indices_u1)
dim_u2 = len(indices_u2)
idx_forecast = [0, 1, 2, 4, 5, 6]

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


##################################################################
############ Visualization for Lead Time Forecast ################
##################################################################
def NRMSE(u, u_pred):
    return torch.mean(torch.sqrt(torch.mean((u-u_pred)**2, dim=0) / torch.var(u, dim=0))).item()

si = int(1020/dt)
ei = int(1050/dt)
test_lead_steps_3m = int(3/12/dt)
test_lead_steps_6m = int(6/12/dt)
test_lead_steps_9m = int(9/12/dt)
test_lead_steps_12m = int(12/12/dt)

cgkn_test_nrmse_lst = np.load("Data/cgkn_test_nrmse_lst.npy")
cgkn_test_corr_lst = np.load("Datacgkn_test_corr_lst.npy")
KoopmanLinear_test_nrmse_lst = np.load("Data/KoopmanLinear_test_nrmse_lst.npy")
KoopmanLinear_test_corr_lst = np.load("Data/KoopmanLinear_test_corr_lst.npy")
reg_test_nrmse_lst = np.load("Data/reg_test_nrmse_lst.npy")
reg_test_corr_lst = np.load("Data/reg_test_corr_lst.npy")
persistence_test_nrmse_lst = []
persistence_test_corr_lst = []
for m in range(1, 13):
    nrmse = NRMSE(test_u[si+int(m/12/dt):ei, 2], test_u[si:ei-int(m/12/dt), 2])
    corr = torch.corrcoef( torch.stack([test_u[si+int(m/12/dt):ei, 2], test_u[si:ei-int(m/12/dt), 2]]) )[0,1].item()
    persistence_test_nrmse_lst.append(nrmse)
    persistence_test_corr_lst.append(corr)
persistence_test_nrmse_lst = np.array(persistence_test_nrmse_lst)
persistence_test_corr_lst = np.array(persistence_test_corr_lst)

cgkn_test_leadPred_3m = np.load("Data/cgkn_3MsLeadForecast(1000Yto1100Y).npy" )
cgkn_test_leadPred_6m = np.load("Data/cgkn_6MsLeadForecast(1000Yto1100Y).npy" )
cgkn_test_leadPred_9m = np.load("Data/cgkn_9MsLeadForecast(1000Yto1100Y).npy" )
cgkn_test_leadPred_12m = np.load("Data/cgkn_12MsLeadForecast(1000Yto1100Y).npy" )
KoopmanLinear_test_leadPred_3m = np.load("Data/KoopmanLinear_3MsLeadForecast(1000Yto1100Y).npy" )
KoopmanLinear_test_leadPred_6m = np.load("Data/KoopmanLinear_6MsLeadForecast(1000Yto1100Y).npy" )
KoopmanLinear_test_leadPred_9m = np.load("Data/KoopmanLinear_9MsLeadForecast(1000Yto1100Y).npy" )
KoopmanLinear_test_leadPred_12m = np.load("Data/KoopmanLinear_12MsLeadForecast(1000Yto1100Y).npy" )
reg_test_leadPred_3m = np.load("Data/reg_3MsLeadForecast(1000Yto1100Y).npy" )
reg_test_leadPred_6m = np.load("Data/reg_6MsLeadForecast(1000Yto1100Y).npy" )
reg_test_leadPred_9m = np.load("Data/reg_9MsLeadForecast(1000Yto1100Y).npy" )
reg_test_leadPred_12m = np.load("Data/reg_12MsLeadForecast(1000Yto1100Y).npy" )
persistence_test_leadPred_3m = test_u[int(1000/dt):int(1100/dt)-test_lead_steps_3m]
persistence_test_leadPred_6m = test_u[int(1000/dt):int(1100/dt)-test_lead_steps_3m]
persistence_test_leadPred_9m = test_u[int(1000/dt):int(1100/dt)-test_lead_steps_3m]
persistence_test_leadPred_12m = test_u[int(1000/dt):int(1100/dt)-test_lead_steps_3m]



fig = plt.figure(figsize=(18, 25))
ax00 = plt.subplot2grid((5, 2), (0, 0))
ax01 = plt.subplot2grid((5, 2), (0, 1))
ax1 = plt.subplot2grid((5, 2), (1, 0), colspan=2)
ax2 = plt.subplot2grid((5, 2), (2, 0), colspan=2)
ax3 = plt.subplot2grid((5, 2), (3, 0), colspan=2)
ax4 = plt.subplot2grid((5, 2), (4, 0), colspan=2)
ax00.plot(range(1, 13), 1 - cgkn_test_nrmse_lst, linewidth=5, marker=".", markersize=30, color="red")
ax00.plot(range(1, 13), 1 - KoopmanLinear_test_nrmse_lst, linewidth=5, marker=".", markersize=30, color="blue")
ax00.plot(range(1, 13), 1 - reg_test_nrmse_lst, linewidth=5, marker=".", markersize=30, color="green")
ax00.plot(range(1, 13), 1 - persistence_test_nrmse_lst, linewidth=5, linestyle="--", marker=".", markersize=30, color="orange")
ax00.grid(True, which='both', axis='both', linestyle='--')
ax00.set_ylabel(r"\textbf{1 - NRMSE}", fontsize=30)
ax00.set_xlabel(r"\textbf{Lead(months)}", fontsize=30)
ax00.set_xlim([0.5, 12.5])
ax00.set_xticks(range(1, 13))
ax00.set_ylim([-0.7, 1.1])
ax00.set_yticks([-0.5, 0., 0.5, 1.])
ax01.plot(range(1, 13), cgkn_test_corr_lst, linewidth=5, marker=".", markersize=30, color="red")
ax01.plot(range(1, 13), KoopmanLinear_test_corr_lst, linewidth=5, marker=".", markersize=30, color="blue")
ax01.plot(range(1, 13), reg_test_corr_lst, linewidth=5, marker=".", markersize=30, color="green")
ax01.plot(range(1, 13), persistence_test_corr_lst, linewidth=5, linestyle="--", marker=".", markersize=30, color="orange")
ax01.grid(True, which='both', axis='both', linestyle='--')
ax01.set_ylabel(r"\textbf{Corr}", fontsize=25)
ax01.set_xlabel(r"\textbf{Lead(months)}", fontsize=25)
ax01.set_xlim([0.5, 12.5])
ax01.set_xticks(range(1, 13))
ax01.set_ylim([-0.3, 1.1])
ax01.set_yticks([0., 0.5, 1.])
ax1.plot(test_t[si:ei], test_u[si:ei, 2], linewidth=5.5,  color="black", label=r"\textbf{True signal}")
ax1.plot(test_t[si:ei], cgkn_test_leadPred_3m[int(20/dt)-test_lead_steps_3m:int(50/dt)-test_lead_steps_3m, 2], linewidth=5,  color="red", label=r"\textbf{CGKN}")
ax1.plot(test_t[si:ei], KoopmanLinear_test_leadPred_3m[int(20/dt)-test_lead_steps_3m:int(50/dt)-test_lead_steps_3m, 2], linewidth=3,  color="blue", label=r"\textbf{Simplified KoopNet}")
ax1.plot(test_t[si:ei], reg_test_leadPred_3m[int(20/dt)-test_lead_steps_3m:int(50/dt)-test_lead_steps_3m, 2], linewidth=2.5,  color="green", label=r"\textbf{CG-Reg}")
ax1.plot(test_t[si:ei], persistence_test_leadPred_3m[int(20/dt)-test_lead_steps_3m:int(50/dt)-test_lead_steps_3m, 2], linewidth=3, linestyle="--", color="orange", label=r"\textbf{Persistence}")
ax2.plot(test_t[si:ei], test_u[si:ei, 2], linewidth=5.5,  color="black")
ax2.plot(test_t[si:ei], cgkn_test_leadPred_6m[int(20/dt)-test_lead_steps_6m:int(50/dt)-test_lead_steps_6m, 2], linewidth=5,  color="red")
ax2.plot(test_t[si:ei], KoopmanLinear_test_leadPred_6m[int(20/dt)-test_lead_steps_6m:int(50/dt)-test_lead_steps_6m, 2], linewidth=3,  color="blue")
ax2.plot(test_t[si:ei], reg_test_leadPred_6m[int(20/dt)-test_lead_steps_6m:int(50/dt)-test_lead_steps_6m, 2], linewidth=2.5,  color="green")
ax2.plot(test_t[si:ei], persistence_test_leadPred_6m[int(20/dt)-test_lead_steps_6m:int(50/dt)-test_lead_steps_6m, 2], linewidth=3, linestyle="--", color="orange")
ax3.plot(test_t[si:ei], test_u[si:ei, 2], linewidth=5.5,  color="black")
ax3.plot(test_t[si:ei], cgkn_test_leadPred_9m[int(20/dt)-test_lead_steps_9m:int(50/dt)-test_lead_steps_9m, 2], linewidth=5,  color="red")
ax3.plot(test_t[si:ei], KoopmanLinear_test_leadPred_9m[int(20/dt)-test_lead_steps_9m:int(50/dt)-test_lead_steps_9m, 2], linewidth=3,  color="blue")
ax3.plot(test_t[si:ei], reg_test_leadPred_9m[int(20/dt)-test_lead_steps_9m:int(50/dt)-test_lead_steps_9m, 2], linewidth=2.5,  color="green")
ax3.plot(test_t[si:ei], persistence_test_leadPred_9m[int(20/dt)-test_lead_steps_9m:int(50/dt)-test_lead_steps_9m, 2], linewidth=3, linestyle="--", color="orange")
ax4.plot(test_t[si:ei], test_u[si:ei, 2], linewidth=5.5,  color="black")
ax4.plot(test_t[si:ei], cgkn_test_leadPred_12m[int(20/dt)-test_lead_steps_12m:int(50/dt)-test_lead_steps_12m, 2], linewidth=5,  color="red")
ax4.plot(test_t[si:ei], KoopmanLinear_test_leadPred_12m[int(20/dt)-test_lead_steps_12m:int(50/dt)-test_lead_steps_12m, 2], linewidth=3,  color="blue")
ax4.plot(test_t[si:ei], reg_test_leadPred_12m[int(20/dt)-test_lead_steps_12m:int(50/dt)-test_lead_steps_12m, 2], linewidth=2.5,  color="green")
ax4.plot(test_t[si:ei], persistence_test_leadPred_12m[int(20/dt)-test_lead_steps_12m:int(50/dt)-test_lead_steps_12m, 2], linewidth=3, linestyle="--", color="orange")
ax4.set_xlabel(r"\textbf{Year}", fontsize=30)
ax00.set_title(r"\textbf{(a) NRMSE}", fontsize=30)
ax01.set_title(r"\textbf{(b) Correlation}", fontsize=30)
ax1.set_title(r"\textbf{(c) Forecast at a lead time of 3 months}", fontsize=30)
ax2.set_title(r"\textbf{(d) Forecast at a lead time of 6 months}", fontsize=30)
ax3.set_title(r"\textbf{(e) Forecast at a lead time of 9 months}", fontsize=30)
ax4.set_title(r"\textbf{(f) Forecast at a lead time of 12 months}", fontsize=30)
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_ylabel(r"$T_E$", fontsize=30)
    ax.set_xlim([test_t[si], test_t[ei]])
    ax.set_xticks(np.arange(1020, 1050+5, 5))
    ax.set_ylim([-3, 4])
    ax.set_yticks([-2, 0, 2, 4])
for ax in fig.get_axes():
    ax.tick_params(labelsize=30, length=15, width=1, direction="in", top=True, bottom=True, left=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
fig.tight_layout()
lege_handle, lege_label = ax1.get_legend_handles_labels()
lege = fig.legend(handles=[lege_handle[i] for i in [0,2,1,4,3]], labels=[lege_label[i] for i in [0,2,1,4,3]], fontsize=30, loc="upper center", ncol=3, fancybox=False, edgecolor="black", bbox_to_anchor=(0.525, 1))
lege.get_frame().set_linewidth(1)
fig.subplots_adjust(top=0.91, wspace=0.3, hspace=0.45)
# fig.savefig("Figs/ENSO_LeadForecast.png")
# fig.savefig("Figs/ENSO_LeadForecast.pdf")


#################################################################
############ Visualization for Data Assimilation ################
#################################################################

test_u2 = test_u[:, 4:]
cgkn_DA_mean = np.load("Data/cgkn_DA_mean.npy")
cgkn_DA_std = np.load("Data/cgkn_DA_std.npy")
# KoopmanLinear_DA_mean = np.load("Data/KoopmanLinear_DA_mean.npy")
# KoopmanLinear_DA_std = np.load("Data/KoopmanLinear_DA_mean.npy")


si = int(1020/dt)
ei = int(1050/dt)
fig = plt.figure(figsize=(18, 14))
axs = fig.subplots(3, 1)
axs[0].plot(test_t[si:ei], test_u2[si:ei, 0], color="black", linewidth=4, label=r"\textbf{True signal}")
axs[0].plot(test_t[si:ei], cgkn_DA_mean[si:ei, 0], color="red", linewidth=2, label=r"\textbf{CGKN}")
axs[0].fill_between(test_t[si:ei], cgkn_DA_mean[si:ei, 0]-2*cgkn_DA_std[si:ei, 0], cgkn_DA_mean[si:ei, 0]+2*cgkn_DA_std[si:ei, 0], color='grey', alpha=1.0, label=r"\textbf{Uncertainty}")
axs[1].plot(test_t[si:ei], test_u2[si:ei, 1], color="black", linewidth=4)
axs[1].plot(test_t[si:ei], cgkn_DA_mean[si:ei, 1], color="red", linewidth=2)
axs[1].fill_between(test_t[si:ei], cgkn_DA_mean[si:ei, 1]-2*cgkn_DA_std[si:ei, 1], cgkn_DA_mean[si:ei, 1]+2*cgkn_DA_std[si:ei, 1], color='grey', alpha=0.8)
axs[2].plot(test_t[si:ei], test_u2[si:ei, 2], color="black", linewidth=4)
axs[2].plot(test_t[si:ei], cgkn_DA_mean[si:ei, 2], color="red", linewidth=2)
axs[2].fill_between(test_t[si:ei], cgkn_DA_mean[si:ei, 2]-2*cgkn_DA_std[si:ei, 2], cgkn_DA_mean[si:ei, 2]+2*cgkn_DA_std[si:ei, 2], color='grey', alpha=0.8)
axs[0].set_ylabel(r"$H_W$", fontsize=30)
axs[1].set_ylabel(r"$H_C$", fontsize=30)
axs[2].set_ylabel(r"$H_E$", fontsize=30)
axs[2].set_xlabel(r"\textbf{Year}", fontsize=30)
for ax in fig.get_axes():
    ax.set_xlim([test_t[si], test_t[ei]])
    ax.set_ylim([-50, 50])
    ax.set_yticks([-40, -20, 0, 20, 40])
    ax.tick_params(labelsize=30, length=18, width=1, direction="in", top=True, bottom=True, left=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
fig.tight_layout()
lege = fig.legend(fontsize=30, loc="upper center", ncol=3, fancybox=False, edgecolor="black", bbox_to_anchor=(0.525, 1)) #
lege.get_frame().set_linewidth(1)
fig.subplots_adjust(top=0.92)
# fig.savefig("Figs/ENSO_DA.png")
# fig.savefig("Figs/ENSO_DA.pdf")





# si = int(1500/dt)
# ei = int(1520/dt)
# fig = plt.figure(figsize=(18, 15))
# axs = fig.subplots(3, 2)
# # CGKN
# axs[0, 0].plot(test_t[si:ei], test_u2[si:ei, 0], color="black", linewidth=4, label=r"\textbf{True signal}")
# axs[0, 0].plot(test_t[si:ei], cgkn_DA_mean[si:ei, 0], color="red", linewidth=2, label=r"\textbf{CGKN}")
# axs[0, 0].fill_between(test_t[si:ei], cgkn_DA_mean[si:ei, 0]-2*cgkn_DA_std[si:ei, 0], cgkn_DA_mean[si:ei, 0]+2*cgkn_DA_std[si:ei, 0], color='grey', alpha=0.8, label=r"\textbf{Uncertainty}")
# axs[1, 0].plot(test_t[si:ei], test_u2[si:ei, 1], color="black", linewidth=4)
# axs[1, 0].plot(test_t[si:ei], cgkn_DA_mean[si:ei, 1], color="red", linewidth=2)
# axs[1, 0].fill_between(test_t[si:ei], cgkn_DA_mean[si:ei, 1]-2*cgkn_DA_std[si:ei, 1], cgkn_DA_mean[si:ei, 1]+2*cgkn_DA_std[si:ei, 1], color='grey', alpha=0.8)
# axs[2, 0].plot(test_t[si:ei], test_u2[si:ei, 2], color="black", linewidth=4)
# axs[2, 0].plot(test_t[si:ei], cgkn_DA_mean[si:ei, 2], color="red", linewidth=2)
# axs[2, 0].fill_between(test_t[si:ei], cgkn_DA_mean[si:ei, 2]-2*cgkn_DA_std[si:ei, 2], cgkn_DA_mean[si:ei, 2]+2*cgkn_DA_std[si:ei, 2], color='grey', alpha=0.8)
# # Simplified KoopNet
# axs[0, 1].plot(test_t[si:ei], test_u2[si:ei, 0], color="black", linewidth=4)
# axs[0, 1].plot(test_t[si:ei], KoopmanLinear_DA_mean[si:ei, 0], color="blue", linewidth=1.5, label=r"\textbf{Simplified KoopNet}")
# axs[0, 1].fill_between(test_t[si:ei], KoopmanLinear_DA_mean[si:ei, 0]-2*cgkn_DA_std[si:ei, 0], KoopmanLinear_DA_mean[si:ei, 0]+2*cgkn_DA_std[si:ei, 0], color='grey', alpha=0.8)
# axs[1, 1].plot(test_t[si:ei], test_u2[si:ei, 1], color="black", linewidth=4)
# axs[1, 1].plot(test_t[si:ei], KoopmanLinear_DA_mean[si:ei, 1], color="blue", linewidth=1.5)
# axs[1, 1].fill_between(test_t[si:ei], KoopmanLinear_DA_mean[si:ei, 1]-2*cgkn_DA_std[si:ei, 1], KoopmanLinear_DA_mean[si:ei, 1]+2*cgkn_DA_std[si:ei, 1], color='grey', alpha=0.8)
# axs[2, 1].plot(test_t[si:ei], test_u2[si:ei, 2], color="black", linewidth=4)
# axs[2, 1].plot(test_t[si:ei], KoopmanLinear_DA_mean[si:ei, 2], color="blue", linewidth=1.5)
# axs[2, 1].fill_between(test_t[si:ei], KoopmanLinear_DA_mean[si:ei, 2]-2*cgkn_DA_std[si:ei, 2], KoopmanLinear_DA_mean[si:ei, 2]+2*cgkn_DA_std[si:ei, 2], color='grey', alpha=0.8)
# axs[0, 0].set_title(r"\textbf{(a) CGKN}", fontsize=25, pad=20)
# axs[0, 1].set_title(r"\textbf{(b) Simplified KoopNet}", fontsize=25, pad=20)
# axs[0, 0].set_ylabel(r"$H_W$", fontsize=30)
# axs[1, 0].set_ylabel(r"$H_C$", fontsize=30)
# axs[2, 0].set_ylabel(r"$H_E$", fontsize=30)
# axs[2, 0].set_xlabel(r"$t$", fontsize=30)
# axs[2, 1].set_xlabel(r"$t$", fontsize=30)
# axs[0,0].set_ylim([-22, 22])
# axs[0,1].set_ylim([-22, 22])
# axs[1,0].set_ylim([-30, 22])
# axs[1,1].set_ylim([-30, 22])
# axs[2,0].set_ylim([-25, 22])
# axs[2,1].set_ylim([-25, 22])
# for ax in fig.get_axes():
#     ax.set_xlim([test_t[si], test_t[ei]])
#     ax.set_yticks(np.arange(-20, 20+10, 10))
#     ax.tick_params(labelsize=25, length=15, width=1, direction="in", top=True, bottom=True, left=True, right=True)
#     for spine in ax.spines.values():
#         spine.set_linewidth(2)
# axs[1,0].set_yticks(np.arange(-20, 20+10, 10))
# axs[1,1].set_yticks(np.arange(-20, 20+10, 10))
# fig.tight_layout()
# lege_handles, lege_labels = axs[0,0].get_legend_handles_labels()
# lege_handles.insert(2, axs[0,1].get_legend_handles_labels()[0][0])
# lege_labels.insert(2, axs[0,1].get_legend_handles_labels()[1][0])
# lege = fig.legend(handles=lege_handles, labels=lege_labels, fontsize=25, loc="upper center", ncol=4, fancybox=False, edgecolor="black", bbox_to_anchor=(0.525, 1)) #
# lege.get_frame().set_linewidth(1)
# fig.subplots_adjust(top=0.9)
# fig.savefig("Figs/ENSO_DA.png")
# fig.savefig("Figs/ENSO_DA.pdf")
