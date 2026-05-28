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
failure_rate_array_cfg = np.load("run_0_failure_rate_array_cfg.npy")
C_f_value_array_cfg_run = np.load("run_0_C_f_value_array_cfg_run.npy")
D_f_value_array_cfg_run = np.load("run_0_D_f_value_array_cfg_run.npy")
S_f_value_array_cfg_run = np.load("run_0_S_f_value_array_cfg_run.npy")
C_f_error_array_cfg_run = np.load("run_0_C_f_error_array_cfg_run.npy")
D_f_error_array_cfg_run = np.load("run_0_D_f_error_array_cfg_run.npy")
S_f_error_array_cfg_run = np.load("run_0_S_f_error_array_cfg_run.npy")
f_sig_value_array_cfg_run = np.load("run_0_f_sig_value_array_cfg_run.npy")
f_sig_error_array_cfg_run = np.load("run_0_f_sig_error_array_cfg_run.npy")





#Check fit failure rate
print("Failure rate for each configuration =", failure_rate_array_cfg)





#Fitted paramter value statistics
#Mean
C_f_mean_array_cfg = np.mean(C_f_value_array_cfg_run, axis=1)
D_f_mean_array_cfg = np.mean(D_f_value_array_cfg_run, axis=1)
S_f_mean_array_cfg = np.mean(S_f_value_array_cfg_run, axis=1)
#Width
C_f_width_array_cfg = np.std(C_f_value_array_cfg_run, axis=1, ddof=1)
D_f_width_array_cfg = np.std(D_f_value_array_cfg_run, axis=1, ddof=1)
S_f_width_array_cfg = np.std(S_f_value_array_cfg_run, axis=1, ddof=1)
#Error in mean
C_f_mean_error_array_cfg = C_f_width_array_cfg/np.sqrt(run_number)
D_f_mean_error_array_cfg = D_f_width_array_cfg/np.sqrt(run_number)
S_f_mean_error_array_cfg = S_f_width_array_cfg/np.sqrt(run_number)
#Error in width
C_f_width_error_array_cfg = C_f_width_array_cfg/np.sqrt(2*(run_number-1))
D_f_width_error_array_cfg = D_f_width_array_cfg/np.sqrt(2*(run_number-1))
S_f_width_error_array_cfg = S_f_width_array_cfg/np.sqrt(2*(run_number-1))

#Fitted parameter pull statistics
#Compute the pull
C_f_pull_array_cfg_run = (C_f_value_array_cfg_run-C_f)/C_f_error_array_cfg_run
D_f_pull_array_cfg_run = (D_f_value_array_cfg_run-D_f)/D_f_error_array_cfg_run
S_f_pull_array_cfg_run = (S_f_value_array_cfg_run-S_f)/S_f_error_array_cfg_run
#Mean
C_f_pull_mean_array_cfg = np.mean(C_f_pull_array_cfg_run, axis=1)
D_f_pull_mean_array_cfg = np.mean(D_f_pull_array_cfg_run, axis=1)
S_f_pull_mean_array_cfg = np.mean(S_f_pull_array_cfg_run, axis=1)
#Width
C_f_pull_width_array_cfg = np.std(C_f_pull_array_cfg_run, axis=1, ddof=1)
D_f_pull_width_array_cfg = np.std(D_f_pull_array_cfg_run, axis=1, ddof=1)
S_f_pull_width_array_cfg = np.std(S_f_pull_array_cfg_run, axis=1, ddof=1)
#Error in mean
C_f_pull_mean_error_array_cfg = C_f_pull_width_array_cfg/np.sqrt(run_number)
D_f_pull_mean_error_array_cfg = D_f_pull_width_array_cfg/np.sqrt(run_number)
S_f_pull_mean_error_array_cfg = S_f_pull_width_array_cfg/np.sqrt(run_number)

#Error in width
C_f_pull_width_error_array_cfg = C_f_pull_width_array_cfg/np.sqrt(2*(run_number-1))
D_f_pull_width_error_array_cfg = D_f_pull_width_array_cfg/np.sqrt(2*(run_number-1))
S_f_pull_width_error_array_cfg = S_f_pull_width_array_cfg/np.sqrt(2*(run_number-1))





#Fitted signal fraction statistics
#Mean
f_sig_mean_array_cfg = np.mean(f_sig_value_array_cfg_run, axis=1)
#Width
f_sig_width_array_cfg = np.std(f_sig_value_array_cfg_run, axis=1, ddof=1)
#Error in mean
f_sig_mean_error_array_cfg = f_sig_width_array_cfg/np.sqrt(run_number)
#Error in width
f_sig_width_error_array_cfg = f_sig_width_array_cfg/np.sqrt(2*(run_number-1))

#Fitted signal fraction pull statistics
#Compute the pull
f_sig_pull_array_cfg_run = (f_sig_value_array_cfg_run - f_sig)/f_sig_error_array_cfg_run

#Mean
f_sig_pull_mean_array_cfg = np.mean(f_sig_pull_array_cfg_run, axis=1)
#Width
f_sig_pull_width_array_cfg = np.std(f_sig_pull_array_cfg_run, axis=1, ddof=1)
#Error in mean
f_sig_pull_mean_error_array_cfg = f_sig_pull_width_array_cfg/np.sqrt(run_number)
#Error in width
f_sig_pull_width_error_array_cfg = f_sig_pull_width_array_cfg/np.sqrt(2*(run_number-1))





#Sample-size-scaled S_f = sin(2*beta) statistics
sin2beta_mean_error_array_cfg = np.sqrt(N_sig/N_sig_expected) * S_f_mean_error_array_cfg
sin2beta_width_array_cfg = np.sqrt(N_sig/N_sig_expected) * S_f_width_array_cfg
sin2beta_width_error_array_cfg = np.sqrt(N_sig/N_sig_expected) * S_f_width_error_array_cfg


#Angle beta statistics
#Compute angle beta
beta = 180/np.pi * 0.5 * np.arcsin(S_f)
beta_value_array_cfg_run = 180/np.pi * 0.5 * np.arcsin(S_f_value_array_cfg_run)
beta_error_array_cfg_run = 180/np.pi * 0.5 * 1/np.sqrt(1-S_f_value_array_cfg_run**2) * S_f_error_array_cfg_run
#Mean
beta_mean_array_cfg = np.mean(beta_value_array_cfg_run, axis=1)
#Width
beta_width_array_cfg = np.std(beta_value_array_cfg_run, axis=1, ddof=1)
#Error in mean
beta_mean_error_array_cfg = beta_width_array_cfg/np.sqrt(run_number)
#Error in width
beta_width_error_array_cfg = beta_width_array_cfg/np.sqrt(2*(run_number-1))

#Scale results to the expected number of initial signals
beta_mean_error_array_cfg = np.sqrt(N_sig/N_sig_expected) * beta_mean_error_array_cfg
beta_width_array_cfg = np.sqrt(N_sig/N_sig_expected) * beta_width_array_cfg
beta_width_error_array_cfg = np.sqrt(N_sig/N_sig_expected) * beta_width_error_array_cfg





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





#Plot fitted parameter value (mean +/- width) vs mistag rate
fig, ax = plt.subplots()
#ax.set_title(r"Fitted $C_{f}$ vs Mistag Rate")
ax.set_xlabel(r"Mistag Rate $\omega$")
ax.set_ylabel(r"$C_{f}$")
ax.errorbar(omega_array_cfg, C_f_mean_array_cfg, yerr=C_f_width_array_cfg, fmt="o", markersize=4, capsize=2, label="Simulation")
ax.plot(np.array([0, 1]), C_f*np.ones(2), color="black", label=rf"True Value $C_{{f}}$ = {C_f}")
ax.set_xlim(0, omega_array_cfg[-1]+0.02)
ax.legend(loc="upper left")
ax.grid(True)
plt.tight_layout()
plt.savefig("C_f_value_vs_mistag_rate.pdf")
plt.show()

fig, ax = plt.subplots()
#ax.set_title(r"Fitted $S_{f}$ vs Mistag Rate")
ax.set_xlabel(r"Mistag Rate $\omega$")
ax.set_ylabel(r"$S_{f}$")
ax.errorbar(omega_array_cfg, S_f_mean_array_cfg, yerr=S_f_width_array_cfg, fmt="o", markersize=4, capsize=2, label="Simulation")
ax.plot(np.array([0, 1]), S_f*np.ones(2), color="black", label=rf"True Value $S_{{f}}$ = {S_f}")
ax.set_xlim(0, omega_array_cfg[-1]+0.02)
#label = [Line2D([0], [0], color="C0", label="Simulation"), Line2D([0], [0], color="black", label=r"True Value $S_{f} = 0.711$")]
#ax.legend(handles=label, loc="upper left")
ax.legend(loc="upper left")
ax.grid(True)
plt.tight_layout()
plt.savefig("S_f_value_vs_mistag_rate.pdf")
plt.show()

fig, ax = plt.subplots()
#ax.set_title(r"Fitted $f_{sig}$ vs Mistag Rate")
ax.set_xlabel(r"Mistag Rate $\omega$")
ax.set_ylabel(r"Signal Fraction $f_{sig}$")
ax.errorbar(omega_array_cfg, f_sig_mean_array_cfg, yerr=f_sig_width_array_cfg, fmt="o", markersize=4, capsize=2, label="Simulation")
ax.plot(np.array([0, 1]), f_sig*np.ones(2), color="black", label=rf"True Value $f_{{sig}}$ = {f_sig}")
ax.set_xlim(0, omega_array_cfg[-1]+0.02)
ax.legend(loc="upper left")
ax.grid(True)
plt.tight_layout()
plt.savefig("f_sig_value_vs_mistag_rate.pdf")
plt.show()





#Plot fitted parameter width (width +/- width_error) vs mistag rate
fig, ax = plt.subplots()
#ax_twin = ax.twinx()
#ax.set_title(r"Standard Deviation of $C_{f}$ vs Mistag Rate")
ax.set_xlabel(r"Mistag Rate $\omega$")
ax.set_ylabel(r"Standard Deviation $\sigma(C_{f})$")
ax.errorbar(omega_array_cfg, C_f_width_array_cfg, yerr=C_f_width_error_array_cfg, fmt="o", markersize=4, capsize=2, label="Simulation")
ax.plot(omega_array_cfg, C_f_width_array_cfg[-1]*(1-2*omega_array_cfg[-1])*1/(1-2*omega_array_cfg), color="black", label=r"Expectation ~1/(1-2$\omega$)")
ax.set_xlim(0, omega_array_cfg[-1]+0.02)
ax.set_ylim(0, )
ax.legend(loc="upper left")
ax.grid(True)
plt.tight_layout()
plt.savefig("C_f_width_vs_mistag_rate.pdf")
plt.show()

fig, ax = plt.subplots()
#ax.set_title(r"Standard Deviation of $S_{f}$ vs Mistag Rate")
ax.set_xlabel(r"Mistag Rate $\omega$")
ax.set_ylabel(r"Standard Deviation $\sigma(S_{f})$")
ax.errorbar(omega_array_cfg, S_f_width_array_cfg, yerr=S_f_width_error_array_cfg, fmt="o", markersize=4, capsize=2, label="Simulation")
ax.plot(omega_array_cfg, S_f_width_array_cfg[-1]*(1-2*omega_array_cfg[-1])*1/(1-2*omega_array_cfg), color="black", label=r"Expectation ~1/(1-2$\omega$)")
ax.set_xlim(0, omega_array_cfg[-1]+0.02)
ax.set_ylim(0, )
ax.legend(loc="upper left")
ax.grid(True)
plt.tight_layout()
plt.savefig("S_f_width_vs_mistag_rate.pdf")
plt.show()

fig, ax = plt.subplots()
#ax.set_title(r"Standard Deviation of $f_{sig}$ vs Mistag Rate")
ax.set_xlabel(r"Mistag Rate $\omega$")
ax.set_ylabel(r"Standard Deviation $\sigma(f_{sig})$")
ax.errorbar(omega_array_cfg, f_sig_width_array_cfg, yerr=f_sig_width_error_array_cfg, fmt="o", markersize=4, capsize=2, label="Simulation")
ax.set_xlim(0, omega_array_cfg[-1]+0.02)
ax.legend(loc="upper left")
ax.grid(True)
plt.tight_layout()
plt.savefig("f_sig_width_vs_mistag_rate.pdf")
plt.show()





#Plot parameter pull mean (mean +/- mean error) and pull width (width +/- width error) vs mistag rate
fig, ax = plt.subplots()
#ax.set_title(r"Pull Mean of $C_{f}$ vs Mistag Rate")
ax.set_xlabel(r"Mistag Rate $\omega$")
ax.set_ylabel(r"Pull Mean of $C_{f}$")
ax.errorbar(omega_array_cfg, C_f_pull_mean_array_cfg, yerr=C_f_pull_mean_error_array_cfg, fmt="o", markersize=4, capsize=2, label="Simulation")
ax.plot(np.array([0, 1]), np.zeros(2), color="black", label=r"Expectation")
ax.set_xlim(0, omega_array_cfg[-1]+0.02)
ymin, ymax = ax.get_ylim()
y_half_range = max(ymax, -ymin)
ax.set_ylim(-y_half_range, y_half_range)
ax.legend(loc="upper left")
ax.grid(True)
plt.tight_layout()
plt.savefig("C_f_pull_mean_vs_mistag_rate.pdf")
plt.show()

fig, ax = plt.subplots()
#ax.set_title(r"Pull Width of $C_{f}$ vs Mistag Rate")
ax.set_xlabel(r"Mistag Rate $\omega$")
ax.set_ylabel(r"Pull Width of $C_{f}$")
ax.errorbar(omega_array_cfg, C_f_pull_width_array_cfg, yerr=C_f_pull_width_error_array_cfg, fmt="o", markersize=4, capsize=2, label="Simulation")
ax.plot(np.array([0, 1]), np.ones(2), color="black", label=r"Expectation")
ax.set_xlim(0, omega_array_cfg[-1]+0.02)
ymin, ymax = ax.get_ylim()
y_half_range = max(ymax-1, 1-ymin)
ax.set_ylim(1-y_half_range, 1+y_half_range)
ax.legend(loc="upper left")
ax.grid(True)
plt.tight_layout()
plt.savefig("C_f_pull_width_vs_mistag_rate.pdf")
plt.show()


fig, ax = plt.subplots()
#ax.set_title(r"Pull Mean of $S_{f}$ vs Mistag Rate")
ax.set_xlabel(r"Mistag Rate $\omega$")
ax.set_ylabel(r"Pull Mean of $S_{f}$")
ax.errorbar(omega_array_cfg, S_f_pull_mean_array_cfg, yerr=S_f_pull_mean_error_array_cfg, fmt="o", markersize=4, capsize=2, label="Simulation")
ax.plot(np.array([0, 1]), np.zeros(2), color="black", label=r"Expectation")
ax.set_xlim(0, omega_array_cfg[-1]+0.02)
ymin, ymax = ax.get_ylim()
y_half_range = max(ymax, -ymin)
ax.set_ylim(-y_half_range, y_half_range)
ax.legend(loc="upper left")
ax.grid(True)
plt.tight_layout()
plt.savefig("S_f_pull_mean_vs_mistag_rate.pdf")
plt.show()

fig, ax = plt.subplots()
#ax.set_title(r"Pull Width of $S_{f}$ vs Mistag Rate")
ax.set_xlabel(r"Mistag Rate $\omega$")
ax.set_ylabel(r"Pull Width of $S_{f}$")
ax.errorbar(omega_array_cfg, S_f_pull_width_array_cfg, yerr=S_f_pull_width_error_array_cfg, fmt="o", markersize=4, capsize=2, label="Simulation")
ax.plot(np.array([0, 1]), np.ones(2), color="black", label=r"Expectation")
ax.set_xlim(0, omega_array_cfg[-1]+0.02)
ymin, ymax = ax.get_ylim()
y_half_range = max(ymax-1, 1-ymin)
ax.set_ylim(1-y_half_range, 1+y_half_range)
ax.legend(loc="upper left")
ax.grid(True)
plt.tight_layout()
plt.savefig("S_f_pull_width_vs_mistag_rate.pdf")
plt.show()


fig, ax = plt.subplots()
#ax.set_title(r"Pull Mean of $f_{sig}$ vs Mistag Rate")
ax.set_xlabel(r"Mistag Rate $\omega$")
ax.set_ylabel(r"Pull Mean of $f_{sig}$")
ax.errorbar(omega_array_cfg, f_sig_pull_mean_array_cfg, yerr=f_sig_pull_mean_error_array_cfg, fmt="o", markersize=4, capsize=2, label="Simulation")
ax.plot(np.array([0, 1]), np.zeros(2), color="black", label=r"Expectation")
ax.set_xlim(0, omega_array_cfg[-1]+0.02)
ymin, ymax = ax.get_ylim()
y_half_range = max(ymax, -ymin)
ax.set_ylim(-y_half_range, y_half_range)
ax.legend(loc="upper left")
ax.grid(True)
plt.tight_layout()
plt.savefig("f_sig_pull_mean_vs_mistag_rate.pdf")
plt.show()

fig, ax = plt.subplots()
#ax.set_title(r"Pull Width of $f_{sig}$ vs Mistag Rate")
ax.set_xlabel(r"Mistag Rate $\omega$")
ax.set_ylabel(r"Pull Width of $f_{sig}$")
ax.errorbar(omega_array_cfg, f_sig_pull_width_array_cfg, yerr=f_sig_pull_width_error_array_cfg, fmt="o", markersize=4, capsize=2, label="Simulation")
ax.plot(np.array([0, 1]), np.ones(2), color="black", label=r"Expectation")
ax.set_xlim(0, omega_array_cfg[-1]+0.02)
ymin, ymax = ax.get_ylim()
y_half_range = max(ymax-1, 1-ymin)
ax.set_ylim(1-y_half_range, 1+y_half_range)
ax.legend(loc="upper left")
ax.grid(True)
plt.tight_layout()
plt.savefig("f_sig_pull_width_vs_mistag_rate.pdf")
plt.show()





"""
#Plot sin(2*beta) width vs mistag rate
fig, ax = plt.subplots()
#ax.set_title(r"Standard Deviation of $\sin(2 \beta)$ vs Mistag Rate")
ax.set_xlabel(r"Mistag Rate $\omega$")
ax.set_ylabel(r"Standard Deviation $\sigma(\sin(2 \beta))$")
ax.errorbar(omega_array_cfg, sin2beta_width_array_cfg, yerr=sin2beta_width_error_array_cfg, fmt="o", markersize=4, capsize=2, label="Simulation")
ax.plot(omega_array_cfg, sin2beta_width_array_cfg[-1]*(1-2*omega_array_cfg[-1])*1/(1-2*omega_array_cfg), color="black", label=r"Expectation ~1/(1-2$\omega$)")
ax.set_xlim(0, omega_array_cfg[-1]+0.02)
ax.set_ylim(0, )
ax.legend(loc="upper left")
ax.grid(True)
plt.tight_layout()
plt.savefig("sin2beta_width_vs_mistag_rate.pdf")
plt.show()


#Plot angle beta width (width +/- width_error) vs mistag rate
fig, ax = plt.subplots()
#ax.set_title(r"Standard Deviation of Angle $\beta$ vs Mistag Rate")
ax.set_xlabel(r"Mistag Rate $\omega$")
ax.set_ylabel(r"Standard Deviation $\sigma(\beta)$ [$ \degree$]")
ax.errorbar(omega_array_cfg, beta_width_array_cfg, yerr=beta_width_error_array_cfg, fmt="o", markersize=4, capsize=2, label="Simulation")
ax.plot(omega_array_cfg, beta_width_array_cfg[-1]*(1-2*omega_array_cfg[-1])*1/(1-2*omega_array_cfg), color="black", label=r"Expectation 1/(1-2$\omega$)")
ax.set_xlim(0, omega_array_cfg[-1]+0.02)
ax.set_ylim(0, )
ax.legend(loc="upper left")
ax.grid(True)
plt.tight_layout()
plt.savefig("beta_width_vs_mistag_rate.pdf")
plt.show()


##Plot angle beta value (mean +/- width) vs mistag rate
#fig, ax = plt.subplots()
##ax.set_title(r"Fitted Angle $\beta$ vs Mistag Rate")
#ax.set_xlabel(r"Mistag Rate $\omega$")
#ax.set_ylabel(r"Angle $\beta$ [$ \degree$]")
#ax.errorbar(omega_array_cfg, beta_mean_array_cfg, yerr=beta_width_array_cfg, fmt="o", markersize=4, capsize=2, label="Simulation")
#ax.plot(np.array([0, 1]), 22.6174577*np.ones(2), linestyle="dashed", color="black", label=r"Global Fit, $\beta = 22.617 \pm 0.447 \degree$")
#ax.plot(np.array([0, 1]), beta*np.ones(2), linestyle="dashed", color="blue", label=r"From $S_{f}$, $\beta = 22.658 \pm 0.448 \degree$")
##ax.axhspan(22.6174577+0.447495083, 22.6174577-0.447495083, color='red', alpha=0.5)
#ax.set_xlim(0, omega_array_cfg[-1]+0.02)
#ax.legend(loc="upper left")
#ax.grid(True)
#plt.tight_layout()
#plt.savefig("beta_value_vs_mistag_rate.pdf")
#plt.show()
"""





#for cfg_index in range(len(omega_array_cfg)):
#    
#    #Plot the pull distributions
#    plot_range = np.linspace(-3, 3, 1000)
#    bin_edges = np.linspace(-3, 3, 20)
#    bin_midpoints = (bin_edges[:-1] - bin_edges[1:])/2
#    S_f_pull_cfg_array_bin, edges = np.histogram(S_f_pull_array_cfg_run[cfg_index], bins=bin_edges)
#    
#    
#    fig, ax = plt.subplots()
#    #ax_twin = ax.twinx()
#    #ax.set_title(r"Pull Distribution of $S_{f}$")
#    ax.set_xlabel(r"Pull of $S_{f}$")
#    ax.set_ylabel("Number of Runs")
#    #ax.hist(S_f_pull_cfg_array_run, bins=bin_edges, label=rf"$S_{{f}}$ Pull: mean = {S_f_pull_mean_array[N_index][p_sig_index][omega_index]:.5f} $\pm$ {S_f_pull_mean_error_array[N_index][p_sig_index][omega_index]:.5f}, width = {S_f_pull_width_array[N_index][p_sig_index][omega_index]:.5f} $\pm$ {S_f_pull_width_error_array[N_index][p_sig_index][omega_index]:.5f}")
#    #ax_twin.plot(plot_range, scipy.stats.norm.pdf(plot_range, 0, 1), label="N(0, 1)", color="black")
#    ax.errorbar(bin_midpoints, S_f_pull_cfg_array_bin, yerr=np.sqrt(S_f_pull_cfg_array_bin))
#    ax_twin.plot(plot_range, scipy.stats.norm.pdf(plot_range, S_f_pull_mean_array_cfg[cfg_index], S_f_pull_width_array_cfg[cfg_index]), label=fr"$\mu$ = {S_f_pull_mean_array_cfg[cfg_index]:3.f}, $\sigma$ = {S_f_pull_width_array_cfg[cfg_index]:3.f}", color="black")
#    ax.legend(loc="upper right")
#    plt.savefig(f"histogram_{cfg_index:01d}.pdf")
#    plt.show()

#fit_info = [
#    f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {m.fval:.1f} / {m.ndof:.0f} = {m.fmin.reduced_chi2:.1f}",
#]
#for p, v, e in zip(m.parameters, m.values, m.errors):
#    fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")
#
#plt.legend(title="\n".join(fit_info), frameon=False)










#Plot vs dilution
dilution_array_cfg = 1 - 2*omega_array_cfg





"""
#Plot parameter pull mean (mean +/- mean error) and pull width (width +/- width error) vs dilution
fig, ax = plt.subplots()
#ax.set_title(r"Pull Mean of $C_{f}$ vs Dilution $D$")
ax.set_xlabel(r"Dilution $D$")
ax.set_ylabel(r"Pull Mean of $C_{f}$")
ax.errorbar(dilution_array_cfg, C_f_pull_mean_array_cfg, yerr=C_f_pull_mean_error_array_cfg, fmt="o", markersize=4, capsize=2, label="Simulation")
ax.plot(np.array([0, 1]), np.zeros(2), color="black", label=r"Expectation")
ax.set_xlim(dilution_array_cfg[-1]-0.02, 1)
ymin, ymax = ax.get_ylim()
y_half_range = max(ymax, -ymin)
ax.set_ylim(-y_half_range, y_half_range)
ax.legend(loc="upper right")
ax.grid(True)
plt.tight_layout()
plt.savefig("C_f_pull_mean_vs_dilution.pdf")
plt.show()

fig, ax = plt.subplots()
#ax.set_title(r"Pull Width of $C_{f}$ vs Dilution $D$")
ax.set_xlabel(r"Dilution $D$")
ax.set_ylabel(r"Pull Width of $C_{f}$")
ax.errorbar(dilution_array_cfg, C_f_pull_width_array_cfg, yerr=C_f_pull_width_error_array_cfg, fmt="o", markersize=4, capsize=2, label="Simulation")
ax.plot(np.array([0, 1]), np.ones(2), color="black", label=r"Expectation")
ax.set_xlim(dilution_array_cfg[-1]-0.02, 1)
ymin, ymax = ax.get_ylim()
y_half_range = max(ymax-1, 1-ymin)
ax.set_ylim(1-y_half_range, 1+y_half_range)
ax.legend(loc="upper right")
ax.grid(True)
plt.tight_layout()
plt.savefig("C_f_pull_width_vs_dilution.pdf")
plt.show()


fig, ax = plt.subplots()
#ax.set_title(r"Pull Mean of $S_{f}$ vs Dilution $D$")
ax.set_xlabel(r"Dilution $D$")
ax.set_ylabel(r"Pull Mean of $S_{f}$")
ax.errorbar(dilution_array_cfg, S_f_pull_mean_array_cfg, yerr=S_f_pull_mean_error_array_cfg, fmt="o", markersize=4, capsize=2, label="Simulation")
ax.plot(np.array([0, 1]), np.zeros(2), color="black", label=r"Expectation")
ax.set_xlim(dilution_array_cfg[-1]-0.02, 1)
ymin, ymax = ax.get_ylim()
y_half_range = max(ymax, -ymin)
ax.set_ylim(-y_half_range, y_half_range)
ax.legend(loc="upper right")
ax.grid(True)
plt.tight_layout()
plt.savefig("S_f_pull_mean_vs_dilution.pdf")
plt.show()

fig, ax = plt.subplots()
#ax.set_title(r"Pull Width of $S_{f}$ vs Dilution $D$")
ax.set_xlabel(r"Dilution $D$")
ax.set_ylabel(r"Pull Width of $S_{f}$")
ax.errorbar(dilution_array_cfg, S_f_pull_width_array_cfg, yerr=S_f_pull_width_error_array_cfg, fmt="o", markersize=4, capsize=2, label="Simulation")
ax.plot(np.array([0, 1]), np.ones(2), color="black", label=r"Expectation")
ax.set_xlim(dilution_array_cfg[-1]-0.02, 1)
ymin, ymax = ax.get_ylim()
y_half_range = max(ymax-1, 1-ymin)
ax.set_ylim(1-y_half_range, 1+y_half_range)
ax.legend(loc="upper right")
ax.grid(True)
plt.tight_layout()
plt.savefig("S_f_pull_width_vs_dilution.pdf")
plt.show()


fig, ax = plt.subplots()
#ax.set_title(r"Pull Mean of $f_{sig}$ vs Dilution $D$")
ax.set_xlabel(r"Dilution $D$")
ax.set_ylabel(r"Pull Mean of $f_{sig}$")
ax.errorbar(dilution_array_cfg, f_sig_pull_mean_array_cfg, yerr=f_sig_pull_mean_error_array_cfg, fmt="o", markersize=4, capsize=2, label="Simulation")
ax.plot(np.array([0, 1]), np.zeros(2), color="black", label=r"Expectation")
ax.set_xlim(dilution_array_cfg[-1]-0.02, 1)
ymin, ymax = ax.get_ylim()
y_half_range = max(ymax, -ymin)
ax.set_ylim(-y_half_range, y_half_range)
ax.legend(loc="upper right")
ax.grid(True)
plt.tight_layout()
plt.savefig("f_sig_pull_mean_vs_dilution.pdf")
plt.show()

fig, ax = plt.subplots()
#ax.set_title(r"Pull Width of $f_{sig}$ vs Dilution $D$")
ax.set_xlabel(r"Dilution $D$")
ax.set_ylabel(r"Pull Width of $f_{sig}$")
ax.errorbar(dilution_array_cfg, f_sig_pull_width_array_cfg, yerr=f_sig_pull_width_error_array_cfg, fmt="o", markersize=4, capsize=2, label="Simulation")
ax.plot(np.array([0, 1]), np.ones(2), color="black", label=r"Expectation")
ax.set_xlim(dilution_array_cfg[-1]-0.02, 1)
ymin, ymax = ax.get_ylim()
y_half_range = max(ymax-1, 1-ymin)
ax.set_ylim(1-y_half_range, 1+y_half_range)
ax.legend(loc="upper right")
ax.grid(True)
plt.tight_layout()
plt.savefig("f_sig_pull_width_vs_dilution.pdf")
plt.show()
"""





"""
import iminuit
def function(omega, a, b):
    function = a*(1/(1-2*omega)+b)
    return function

def chi2(a, b,):
    chi2 = np.sum(((sin2beta_width_array_cfg - function(omega_array_cfg, a, b))/sin2beta_width_error_array_cfg)**2)
    return chi2

#Fitting error scaling directly to 1/D
mi = iminuit.Minuit(chi2, a=1, b=0)
mi.fixed["b"] = True
mi.migrad()
mi.hesse()
k = mi.values["a"]

#Fitting error scaling to the modified model a*(1/(1-2*b*omega)+c)
mi = iminuit.Minuit(chi2, a=1, b=1, c=0)
mi.migrad()
mi.hesse()
a = mi.values["a"]
b = mi.values["b"]


#Plot sin(2*beta) width vs mistag
fig, ax = plt.subplots()
#ax.set_title(r"Standard Deviation of $\sin(2 \beta)$ vs Mistag Rate $\omega$")
ax.set_xlabel(r"Mistag Rate $\omega$")
ax.set_ylabel(r"Standard Deviation $\sigma(\sin(2 \beta))$")
ax.errorbar(omega_array_cfg, sin2beta_width_array_cfg, yerr=sin2beta_width_error_array_cfg, fmt="o", markersize=4, capsize=2, label="Simulation")
ax.plot(omega_array_cfg, k/(1-2*b1*omega_array_cfg), color="black", linestyle="dashed", alpha=0.5, label="Expectation ~1/D")
ax.plot(omega_array_cfg, a*(1/(1-2*omega_array_cfg)+b), color="black", label="Expectation ~1/D (modified)")
ax.set_xlim(0, omega_array_cfg[-1]+0.02)
ax.set_ylim(0, )
ax.legend(loc="upper right")
ax.grid(True)
plt.tight_layout()
plt.savefig("sin2beta_width_vs_mistag_rate.pdf")
plt.show()


#Plot sin(2*beta) width vs dilution
fig, ax = plt.subplots()
#ax.set_title(r"Standard Deviation of $\sin(2 \beta)$ vs Dilution $D$")
ax.set_xlabel(r"Dilution $D$")
ax.set_ylabel(r"Standard Deviation $\sigma(\sin(2 \beta))$")
ax.errorbar(dilution_array_cfg, sin2beta_width_array_cfg, yerr=sin2beta_width_error_array_cfg, fmt="o", markersize=4, capsize=2, label="Simulation")
ax.plot(dilution_array_cfg, k/(1-2*omega_array_cfg), color="black", linestyle="dashed", alpha=0.5, label="Expectation ~1/D")
ax.plot(dilution_array_cfg, a*(1/(1-2*omega_array_cfg)+b), color="black", label="Expectation ~1/D (modified)")
ax.set_xlim(dilution_array_cfg[-1]-0.02, 1)
ax.set_ylim(0, )
ax.legend(loc="upper right")
ax.grid(True)
plt.tight_layout()
plt.savefig("sin2beta_width_vs_dilution.pdf")
plt.show()


#Plot vs tagging power, assuming 100% tagging efficiency
tagging_power_array_cfg = 1 * (1 - 2*omega_array_cfg)**2

fig, ax = plt.subplots()
#ax.set_title(r"Standard Deviation of $\sin(2 \beta)$ vs Tagging Power $\epsilon_{eff}$")
ax.set_xlabel(r"Tagging Power $\epsilon_{eff}$")
ax.set_ylabel(r"Standard Deviation $\sigma(\sin(2 \beta))$")
ax.errorbar(tagging_power_array_cfg, sin2beta_width_array_cfg, yerr=sin2beta_width_error_array_cfg, fmt="o", markersize=4, capsize=2, label="Simulation")
ax.plot(tagging_power_array_cfg, k/(1-2*omega_array_cfg), color="black", linestyle="dashed", alpha=0.5, label="Expectation ~1/D")
ax.plot(tagging_power_array_cfg, a*1/(1-2*omega_array_cfg), color="black", label="Expectation ~1/D (modified)")
ax.set_xlim(tagging_power_array_cfg[-1]-0.02, 1)
ax.set_ylim(0, )
ax.legend(loc="upper right")
ax.grid(True)
plt.tight_layout()
plt.savefig("sin2beta_width_vs_tagging_power.pdf")
plt.show()
"""





tagging_power_array_cfg = 1 * (1 - 2*omega_array_cfg)**2

import iminuit
def function(tagging_power, a, b):
    function = a*tagging_power**(-0.5) + b
    return function

def chi2(a, b):
    chi2 = np.sum(((sin2beta_width_array_cfg - function(tagging_power_array_cfg, a, b))/sin2beta_width_error_array_cfg)**2)
    return chi2

#Fitting error scaling directly to k/D
mi = iminuit.Minuit(chi2, a=1, b=0)
mi.fixed["b"] = True
mi.migrad()
mi.hesse()
k = mi.values["a"]

#Fitting error scaling to the modified model A/D+B
mi = iminuit.Minuit(chi2, a=1, b=1)
mi.migrad()
mi.hesse()
A = mi.values["a"]
B = mi.values["b"]

#Plot vs tagging power, assuming 100% tagging efficiency
fig, ax = plt.subplots()
#ax.set_title(r"Standard Deviation of $\sin(2 \beta)$ vs Tagging Power $\epsilon_{eff}$")
ax.set_xlabel(r"Tagging Power $\epsilon_{eff}$")
ax.set_ylabel(r"Standard Deviation $\sigma(\sin(2 \beta))$")
ax.errorbar(tagging_power_array_cfg, sin2beta_width_array_cfg, yerr=sin2beta_width_error_array_cfg, fmt="o", markersize=4, capsize=2, label="Simulation")
ax.plot(tagging_power_array_cfg, k*tagging_power_array_cfg**(-0.5), color="black", label=r"Expectation $\sigma(S_{f})=k\varepsilon_{eff}^{-1/2}$")
#ax.plot(tagging_power_array_cfg, A*tagging_power_array_cfg**(-0.5)+B, color="black", label=r"$\sigma(S_{f})=A\varepsilon_{eff}^{-1/2}+B$")
ax.set_xlim(tagging_power_array_cfg[-1]-0.02, 1)
ax.set_ylim(0, )
ax.legend(loc="upper right")
ax.grid(True)
plt.tight_layout()
plt.savefig("sin2beta_width_vs_tagging_power.pdf")
plt.show()