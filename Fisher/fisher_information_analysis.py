import json
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D





#Load configuration
with open("run_0_configuration.json") as f:
    cfg_dict = json.load(f)

#Unpack configuration

#Retrieve mixing and decay parameters
modulus_q_p = cfg_dict["modulus_q_p"]
x = cfg_dict["x"]
y = cfg_dict["y"]
C_f = cfg_dict["C_f"]
S_f = cfg_dict["S_f"]
D_f = cfg_dict["D_f"]

#Retrieve other decay parameters
m_X_f_mean = cfg_dict["m_X_f_mean"]
m_X_f_width = cfg_dict["m_X_f_width"]
m_Xb_f_mean = cfg_dict["m_Xb_f_mean"]
m_Xb_f_width = cfg_dict["m_Xb_f_width"]
tau_bkg = cfg_dict["tau_bkg"]                       #Need to adjust t_range in PDF_mixed generation accordingly
m_bkg_mean = cfg_dict["m_bkg_mean"]
m_bkg_width = cfg_dict["m_bkg_width"]               #Need to adjust base probability outside uniform distribution range

#Retrieve detector, reconstruction, and selection parameters
N_sig_expected = cfg_dict["N_sig_expected"]         #N_sig is a run parameter
f_sig = cfg_dict["f_sig"]
omega = cfg_dict["omega"]                           #omega is a run parameter
eps_X_f = cfg_dict["eps_X_f"]
eps_Xb_f = cfg_dict["eps_Xb_f"]

#Retrieve run parameters
run_number = cfg_dict["run_number"]
N_sig = cfg_dict["N_sig"]
omega_array_cfg = np.array(cfg_dict["omega_list_cfg"])


#Load results
S_f_value_array_cfg_run = np.load("run_0_S_f_value_array_cfg_run.npy")
S_f_error_array_cfg_run = np.load("run_0_S_f_error_array_cfg_run.npy")


#Load fisher information
per_event_fisher_information_S_f_value_array_cfg = np.load("per_event_fisher_information_S_f_value_array_cfg.npy")
per_event_fisher_information_S_f_error_array_cfg = np.load("per_event_fisher_information_S_f_error_array_cfg.npy")





#Fitted paramter value statistics
#Mean
S_f_mean_array_cfg = np.mean(S_f_value_array_cfg_run, axis=1)
#Width
S_f_width_array_cfg = np.std(S_f_value_array_cfg_run, axis=1, ddof=1)
#Error in mean
S_f_mean_error_array_cfg = S_f_width_array_cfg/np.sqrt(run_number)
#Error in width

S_f_width_error_array_cfg = S_f_width_array_cfg/np.sqrt(2*(run_number-1))

#Sample-size-scaled S_f = sin(2*beta) statistics
sin2beta_mean_error_array_cfg = np.sqrt(N_sig/N_sig_expected) * S_f_mean_error_array_cfg
sin2beta_width_array_cfg = np.sqrt(N_sig/N_sig_expected) * S_f_width_array_cfg
sin2beta_width_error_array_cfg = np.sqrt(N_sig/N_sig_expected) * S_f_width_error_array_cfg





#Compute the expected width in the fitted S_f parameter
eps_sig = eps_X_f
N_acc = (N_sig_expected * eps_sig)/f_sig
S_f_width_expected_array_cfg = 1/np.sqrt(N_acc * per_event_fisher_information_S_f_value_array_cfg)
S_f_width_error_expected_array_cfg = 1/2 * 1/np.sqrt(N_acc * per_event_fisher_information_S_f_value_array_cfg**3) * per_event_fisher_information_S_f_error_array_cfg





#Define plot parameters
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": True,
})





"""
#Plot sin(2*beta) width vs mistag rate for simulation estimation and approximate analytic prediction
fig, ax = plt.subplots()
ax.set_title(r"Predicted Standard Deviation of $\sigma(\sin(2 \beta))$ vs Mistag Rate")
ax.set_xlabel(r"Mistag Rate $\omega$")
ax.set_ylabel(r"Standard Deviation $\sigma(\sin(2 \beta))$")
ax.errorbar(omega_array_cfg, sin2beta_width_array_cfg, yerr=sin2beta_width_error_array_cfg, fmt="o", markersize=4, capsize=2, label="Simulation Result")
ax.plot(omega_array_cfg, sin2beta_width_array_cfg[-1]*(1-2*omega_array_cfg[-1])*1/(1-2*omega_array_cfg), color="black")
ax.errorbar(omega_array_cfg, S_f_width_expected_array_cfg, yerr=S_f_width_error_expected_array_cfg, fmt="o", markersize=4, capsize=2, label="Approximate Prediction")
ax.plot(omega_array_cfg, S_f_width_expected_array_cfg[-1] * (1-2*omega_array_cfg[-1]) * 1/(1-2*omega_array_cfg), color="black")
ax.set_xlim(0, omega_array_cfg[-1]+0.02)
ax.set_ylim(0, )
ax.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig("sin2beta_width_vs_mistag_rate_fisher_information.pdf")
plt.show()
"""












##Plot the per-event Fisher information of S_f vs the mistag rate
#fig, ax = plt.subplots()
#ax.set_title(r"Per-Event Fisher Information of $S_{f}$ vs Mistag Rate")
#ax.set_xlabel(r"Mistag Rate $\omega$")
#ax.set_ylabel(r"Per-Event Fisher Information of $S_{f}$")
#ax.errorbar(omega_array_cfg, fisher_information_S_f_value_array_cfg, yerr=fisher_information_S_f_error_array_cfg, fmt="o", markersize=4, capsize=4)
#ax.plot(omega_array_cfg, 1/(N_acc*S_f_width_expected_array_cfg[0]**2) * 1/(1-2*omega_array_cfg[0])**2 * (1-2*omega_array_cfg)**2, color="black", label=r"Expection ~1/(1-2$\omega$)")
#plt.grid(True)
#plt.tight_layout()
#plt.savefig("S_f_per_event_fisher_information_vs_mistag_rate.pdf")
#plt.show()







import iminuit
def function(omega, a, b):
    function = a/(1-2*omega) + b
    return function

def chi2(a, b):
    chi2 = np.sum(((sin2beta_width_array_cfg - function(omega_array_cfg, a, b))/sin2beta_width_error_array_cfg)**2)
    return chi2

#Fitting error scaling for simulation results
mi = iminuit.Minuit(chi2, a=1, b=0)
mi.fixed["b"] = True
mi.migrad()
mi.hesse()
A = mi.values["a"]
B = mi.values["b"]





tagging_power_array_cfg = 1 * (1 - 2*omega_array_cfg)**2

#Plot sin(2*beta) width vs dilution
fig, ax = plt.subplots()
#ax.set_title(r"Standard Deviation of $\sin(2 \beta)$ vs Tagging Power $\varepsilon_{\text{eff}}$")
ax.set_xlabel(r"Tagging Power $\varepsilon_{\text{eff}}$")
ax.set_ylabel(r"Standard Deviation $\sigma(\sin(2 \beta))$")
ax.errorbar(tagging_power_array_cfg, sin2beta_width_array_cfg, yerr=sin2beta_width_error_array_cfg, fmt="o", markersize=4, capsize=2, color="blue", label="Simulation Result")
ax.errorbar(tagging_power_array_cfg, S_f_width_expected_array_cfg, fmt="o", markersize=4, color="red", label="Fisher Information Prediction")
ax.plot(np.linspace(0, 1, 100), A*np.linspace(0, 1, 100)**(-0.5), color="black", label=r"Simulation Fit $\sigma(S_{f})=A\varepsilon_{\text{eff}}^{-1/2}$")
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.003)
ax.legend(loc="upper right")
ax.grid(True)
plt.tight_layout()
plt.savefig("sin2beta_width_vs_tagging_power_fisher_information.pdf")
plt.show()
