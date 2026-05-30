import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import iminuit
from iminuit import Minuit, cost





#Define PDF of decay time (Gamma*t for signal PDFs)
def PDF_t_X_f(t, x, y, C_f, D_f, S_f):
    factor_X_f = 1/(1-y**2) * (1+y*D_f)
    factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
    PDF_t_X_f = 1/(factor_X_f + factor_Y_f) * np.exp(-t) * (np.cosh(y*t)+D_f*np.sinh(y*t)+C_f*np.cos(x*t)-S_f*np.sin(x*t))
    return np.where(t>=0, PDF_t_X_f, 0)

def PDF_t_Xb_f(t, x, y, C_f, D_f, S_f):
    factor_X_f = 1/(1-y**2) * (1+y*D_f)
    factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
    PDF_t_Xb_f = 1/(factor_X_f - factor_Y_f) * np.exp(-t) * (np.cosh(y*t)+D_f*np.sinh(y*t)-C_f*np.cos(x*t)+S_f*np.sin(x*t))
    return np.where(t>=0, PDF_t_Xb_f, 0)

def PDF_t_bkg(t, tau_bkg):
    PDF_t_bkg = 1/tau_bkg * np.exp(-t/tau_bkg)
    return np.where(t>=0, PDF_t_bkg, 0)

#Define PDF of invariant mass m
def PDF_m_X_f(m, m_X_f_mean, m_X_f_width):
    PDF_m_X_f = 1/(np.sqrt(2*np.pi)*m_X_f_width) * np.exp(-((m-m_X_f_mean)**2)/(2*m_X_f_width**2))
    return np.where(m>=0, PDF_m_X_f, 0)

def PDF_m_Xb_f(m, m_Xb_f_mean, m_Xb_f_width):
    PDF_m_Xb_f = 1/(np.sqrt(2*np.pi)*m_Xb_f_width) * np.exp(-((m-m_Xb_f_mean)**2)/(2*m_Xb_f_width**2))
    return np.where(m>=0, PDF_m_Xb_f, 0)

def PDF_m_bkg(m, m_bkg_mean, m_bkg_width):
    PDF_m_bkg = np.zeros(len(m))
    PDF_m_bkg[(m>=(m_bkg_mean-0.5*m_bkg_width)) & (m<=(m_bkg_mean+0.5*m_bkg_width))] = 1/m_bkg_width
    return np.where(m>=0, PDF_m_bkg, 0)


#Define CDFs for use in event generation
def CDF_t_X_f(t, x, y, C_f, D_f, S_f):
    factor_X_f = 1/(1-y**2) * (1+y*D_f)
    factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
    CDF_t_X_f = - 1/(factor_X_f + factor_Y_f) * ( np.exp(-t) * (1/(1-y**2)*((1+y*D_f)*np.cosh(y*t)+(y+D_f)*np.sinh(y*t)) + 1/(1+x**2)*((C_f-x*S_f)*np.cos(x*t)+(-x*C_f-S_f)*np.sin(x*t))) - (1/(1-y**2)*(1+y*D_f) + 1/(1+x**2)*(C_f-x*S_f)) )
    return np.where(t>=0, CDF_t_X_f, 0)

def CDF_t_Xb_f(t, x, y, C_f, D_f, S_f):
    factor_X_f = 1/(1-y**2) * (1+y*D_f)
    factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
    CDF_t_Xb_f = - 1/(factor_X_f - factor_Y_f) * ( np.exp(-t) * (1/(1-y**2)*((1+y*D_f)*np.cosh(y*t)+(y+D_f)*np.sinh(y*t)) + 1/(1+x**2)*((-C_f+x*S_f)*np.cos(x*t)+(x*C_f+S_f)*np.sin(x*t))) - (1/(1-y**2)*(1+y*D_f) + 1/(1+x**2)*(-C_f+x*S_f)) )
    return np.where(t>=0, CDF_t_Xb_f, 0)





#Define the sampling function that returns samples of decay time, invariant mass, and tag
def sample_generation_simple():

    #Compute the ratio of branching fractions
    factor_X_f = 1/(1-y**2) * (1+y*D_f)
    factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
    BF_ratio_Xb_f_X_f = 1/(modulus_q_p**2) * (factor_X_f - factor_Y_f)/(factor_X_f + factor_Y_f)
    #Compute numbers of true X->f and Xb->f signal events
    N_X_f = np.random.poisson(1/(1+BF_ratio_Xb_f_X_f) * N_sig)
    N_Xb_f = np.random.poisson(BF_ratio_Xb_f_X_f/(1+BF_ratio_Xb_f_X_f) * N_sig)
    #Compute the number of background events in the accepted sample
    N_bkg = np.random.poisson((1-f_sig)/f_sig * (eps_X_f*1/(1+BF_ratio_Xb_f_X_f) + eps_Xb_f*BF_ratio_Xb_f_X_f/(1+BF_ratio_Xb_f_X_f)) * N_sig)

    #Generate decay time data for signal and background events
    #Using a sample from a uniform distribution between 0 and 1 as the argument of the inverse CDF gives a sample from the corresponding probability distribution
    #For signal events
    u_t = np.random.uniform(0, 1, N_X_f)
    v_t = np.random.uniform(0, 1, N_Xb_f)
    t_range = np.linspace(0, 100, 100000)
    sample_t_X_f = np.interp(u_t, CDF_t_X_f(t_range, x, y, C_f, D_f, S_f), t_range)
    sample_t_Xb_f = np.interp(v_t, CDF_t_Xb_f(t_range, x, y, C_f, D_f, S_f), t_range)
    #For background events
    if N_bkg == 0:
        sample_t_bkg = np.array([])
    if N_bkg > 0:
        sample_t_bkg = np.random.exponential(tau_bkg, N_bkg)

    #Generate invariant mass data for signals and backgrounds
    sample_m_X_f = np.random.normal(m_X_f_mean, m_X_f_width, N_X_f)
    sample_m_Xb_f = np.random.normal(m_Xb_f_mean, m_Xb_f_width, N_Xb_f)
    if N_bkg == 0:
        sample_m_bkg = np.array([])
    if N_bkg > 0:
        sample_m_bkg = np.random.uniform(m_bkg_mean-0.5*m_bkg_width, m_bkg_mean+0.5*m_bkg_width, N_bkg)

    #Incorporate acceptance of true signal events
    #Acceptance does not apply to background events, as they are only defined in the analysis sample and are not well defined prior to selection
    #Assume an X/Xb->f/bkg event occuring at decay time t has a probability eps_X_f/Xb_f/bkg(t) to be accepted
    u_acc = np.random.uniform(0, 1, N_X_f)
    v_acc = np.random.uniform(0, 1, N_Xb_f)
    #If u/v/w <= eps(t), the event is accepted; if u/v/w > eps(t), the event is not accepted (if eps=1, all events are accepted)
    #Compute the accepted event indices
    acceptance_X_f = (u_acc <= eps_X_f)
    acceptance_Xb_f = (v_acc <= eps_Xb_f)
    #Retain only the accepted signal events
    sample_acc_t_X_f = sample_t_X_f[acceptance_X_f]
    sample_acc_t_Xb_f = sample_t_Xb_f[acceptance_Xb_f]
    sample_acc_m_X_f = sample_m_X_f[acceptance_X_f]
    sample_acc_m_Xb_f = sample_m_Xb_f[acceptance_Xb_f]
    N_acc_X_f = np.sum(acceptance_X_f)
    N_acc_Xb_f = np.sum(acceptance_Xb_f)

    #Incorporate mistagging
    #Assume an X/Xb->f signal event has a probability (1-omega) to be correctly tagged and a probability omega to be mistagged
    #Assuming no biase in the tag assignment of background events (random tag assignment), mistagging does not affect the tagging statistics of background events
    u_tag = np.random.uniform(0, 1, N_acc_X_f)
    v_tag = np.random.uniform(0, 1, N_acc_Xb_f)
    #If u/v < omega, the tag is incorrect; if u/v >= omega, the tag is correct (if omega=0, all tags are correctly assigned)
    #Tagging label for (X, Xbar) is (1, -1)
    #Order decay time events into two samples based on the generated tags
    sample_acc_tag_t_X_f = np.concatenate((sample_acc_t_X_f[u_tag >= omega], sample_acc_t_Xb_f[v_tag < omega]))
    sample_acc_tag_t_Xb_f = np.concatenate((sample_acc_t_Xb_f[v_tag >= omega], sample_acc_t_X_f[u_tag < omega]))
    sample_acc_tag_m_X_f = np.concatenate((sample_acc_m_X_f[u_tag >= omega], sample_acc_m_Xb_f[v_tag < omega]))
    sample_acc_tag_m_Xb_f = np.concatenate((sample_acc_m_Xb_f[v_tag >= omega], sample_acc_m_X_f[u_tag < omega]))
    N_acc_tag_X_f = np.sum(u_tag >= omega) + np.sum(v_tag < omega)
    N_acc_tag_Xb_f = np.sum(v_tag >= omega) + np.sum(u_tag < omega)

    #Construct observed sample: decay time, invariant mass, tag
    sample_t = np.concatenate((sample_acc_tag_t_X_f, sample_acc_tag_t_Xb_f, sample_t_bkg))
    sample_m = np.concatenate((sample_acc_tag_m_X_f, sample_acc_tag_m_Xb_f, sample_m_bkg))
    sample_q = np.concatenate((np.ones(N_acc_tag_X_f), -np.ones(N_acc_tag_Xb_f), np.random.choice([1, -1], size=N_bkg, p=[0.5, 0.5])))

    return sample_t, sample_m, sample_q


#Define the effective PDF of a dataset (t, m, q)
def PDF_simple(t, m, q, C_f, D_f, S_f, f_sig):
    #Define Kronecker delta for tag selection
    delta_q_X_f = np.where(q == 1, 1, 0)
    delta_q_Xb_f = np.where(q == -1, 1, 0)
    #Compute ratio of branching fractions
    factor_X_f = 1/(1-y**2) * (1+y*D_f)
    factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
    BF_ratio_Xb_f_X_f = 1/(modulus_q_p**2) * (factor_X_f - factor_Y_f)/(factor_X_f + factor_Y_f)
    #Compute the probabilities of true signal decay modes P(X/Xb->f)
    P_X_f = f_sig/eps_X_f * 1/(1+BF_ratio_Xb_f_X_f)
    P_Xb_f = f_sig/eps_Xb_f * BF_ratio_Xb_f_X_f/(1+BF_ratio_Xb_f_X_f)
    #Compute the probability of background modes P(bkg) in the accepted sample
    P_bkg = 1 - f_sig
    #Compute conditional tagging probabilities given the true source P(q=X/Xb|X/Xb/bkg)
    #Define P_A_q_B as the probability of A being tagged as q=B
    P_X_f_q_X_f = 1 - omega
    P_X_f_q_Xb_f = omega
    P_Xb_f_q_X_f = omega
    P_Xb_f_q_Xb_f = 1 - omega
    P_bkg_q_X_f = 0.5
    P_bkg_q_Xb_f = 0.5
    #Compute sample PDFs of decay time and invariant mass given the true source
    P_t_X_f = PDF_t_X_f(t, x, y, C_f, D_f, S_f)
    P_t_Xb_f = PDF_t_Xb_f(t, x, y, C_f, D_f, S_f)
    P_m_X_f = PDF_m_X_f(m, m_X_f_mean, m_X_f_width)
    P_m_Xb_f = PDF_m_Xb_f(m, m_Xb_f_mean, m_Xb_f_width)
    P_t_bkg = PDF_t_bkg(t, tau_bkg)
    P_m_bkg = PDF_m_bkg(m, m_bkg_mean, m_bkg_width)
    #Compute the overall effective PDF
    PDF_X_f = P_t_X_f*P_m_X_f*P_X_f_q_X_f*eps_X_f*P_X_f + P_t_Xb_f*P_m_Xb_f*P_Xb_f_q_X_f*eps_Xb_f*P_Xb_f + P_t_bkg*P_m_bkg*P_bkg_q_X_f*P_bkg
    PDF_Xb_f = P_t_X_f*P_m_X_f*P_X_f_q_Xb_f*eps_X_f*P_X_f + P_t_Xb_f*P_m_Xb_f*P_Xb_f_q_Xb_f*eps_Xb_f*P_Xb_f + P_t_bkg*P_m_bkg*P_bkg_q_Xb_f*P_bkg
    PDF_simple = PDF_X_f*delta_q_X_f + PDF_Xb_f*delta_q_Xb_f
    return PDF_simple


#Define the unbinned NLL cost function
#Note that iminuit fits all the arguments of the cost function; sample data and parameters not meant to be fitted must be read globally
def cost_function_simple(C_f, D_f, S_f, f_sig):
    #PDF_sample_simple = np.clip(PDF_simple(sample_t, sample_m, sample_q, C_f, D_f, S_f, f_sig), 1e-300, None)
    PDF_sample_simple = PDF_simple(sample_t, sample_m, sample_q, C_f, D_f, S_f, f_sig)
    cost_function_simple = - np.sum(np.log(PDF_sample_simple))
    return cost_function_simple
#Inform iminuit the cost function is in the NLL form
cost_function_simple.errordef = Minuit.LIKELIHOOD


def A_CP(t, x, y, C_f, D_f, S_f):
    factor_X_f = 1/(1-y**2) * (1+y*D_f)
    factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
    BF_ratio_Xb_f_X_f = 1/(modulus_q_p**2) * (factor_X_f - factor_Y_f)/(factor_X_f + factor_Y_f)
    A_CP = (- PDF_t_X_f(t, x, y, C_f, D_f, S_f) + BF_ratio_Xb_f_X_f*PDF_t_Xb_f(t, x, y, C_f, D_f, S_f))/(PDF_t_X_f(t, x, y, C_f, D_f, S_f) + BF_ratio_Xb_f_X_f*PDF_t_Xb_f(t, x, y, C_f, D_f, S_f))
    return A_CP





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


#Define the mixing and decay parameters
modulus_q_p = 1.0010
x = 0.7697
y = - 0.0005
C_f = 0.002
S_f = 0.711
D_f = - np.sqrt(1 - C_f**2 - S_f**2)

#Plot decay time PDFs and CDFs
t = np.linspace(0, 5, 1000)
fig, ax = plt.subplots()
ax_twin = ax.twinx()
#ax.set_title(r"Decay Time Distribution of $B^{0}/\bar{B}^{0} \rightarrow J/\psi K_{S}^{0}$")
ax.set_xlabel(r"Decay Time $\Gamma t$")
ax.set_ylabel("Probability Density (PDF)")
ax_twin.set_ylabel("Cumulative Probability (CDF)")
ax.plot(t, PDF_t_X_f(t, x, y, C_f, D_f, S_f), color="blue")
ax_twin.plot(t, CDF_t_X_f(t, x, y, C_f, D_f, S_f), color="blue", linestyle=(0,(5,5)))
ax.plot(t, PDF_t_Xb_f(t, x, y, C_f, D_f, S_f), color="red")
ax_twin.plot(t, CDF_t_Xb_f(t, x, y, C_f, D_f, S_f), color="red", linestyle=(5,(5,5)))
ax.set_xlim(0, 5)
ax.set_ylim(0, 1.6)
ax_twin.set_ylim(0, 1)
label = [Line2D([0], [0], color="blue", label=r"$B^{0} \rightarrow J/\psi K_{S}^{0}$"), Line2D([0], [0], color="red", label=r"$\bar{B}^{0} \rightarrow J/\psi K_{S}^{0}$"), Line2D([0], [0], color="black", linestyle="solid", label="PDF"), Line2D([0], [0], color="black", linestyle="dashed", label="CDF")]
plt.legend(handles=label, loc="right")
plt.savefig("decay_time_distribution.pdf")
plt.show()





#Define configuration
modulus_q_p = 1.0010
x = 0.7697
y = - 0.0005
C_f = 0.002
S_f = 0.711
D_f = - np.sqrt(1 - C_f**2 - S_f**2)

m_sig_mean = 5280
m_sig_width = 30
m_X_f_mean = m_sig_mean
m_X_f_width = m_sig_width
m_Xb_f_mean = m_sig_mean
m_Xb_f_width = m_sig_width
tau_bkg = np.nan
m_bkg_mean = np.nan
m_bkg_width = np.nan

N_sig = 10**5
f_sig = 1
omega = 0
eps_X_f = 1
eps_Xb_f = 1

factor_X_f = 1/(1-y**2) * (1+y*D_f)
factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
BF_ratio_Xb_f_X_f = 1/(modulus_q_p**2) * (factor_X_f - factor_Y_f)/(factor_X_f + factor_Y_f)


#Generate samples
sample_t, sample_m, sample_q = sample_generation_simple()

#Bin the samples
t_plot_range = np.linspace(0, 5, 1000)
bin_edges = np.linspace(0, 5, 51)
bin_midpoints = (bin_edges[:-1] + bin_edges[1:])/2
N_X_f_t, edge_X_f = np.histogram(sample_t[sample_q == 1], bins=bin_edges)
N_Xb_f_t, edge_Xb_f = np.histogram(sample_t[sample_q == -1], bins=bin_edges)
A_CP_t = (N_Xb_f_t-N_X_f_t)/(N_X_f_t+N_Xb_f_t)

#Compute pull statistics
N_X_f_t_mean = N_sig*1/(1+BF_ratio_Xb_f_X_f) * (CDF_t_X_f(bin_edges[1:], x, y, C_f, D_f, S_f) - CDF_t_X_f(bin_edges[:-1], x, y, C_f, D_f, S_f))
N_Xb_f_t_mean = N_sig*BF_ratio_Xb_f_X_f/(1+BF_ratio_Xb_f_X_f) * (CDF_t_Xb_f(bin_edges[1:], x, y, C_f, D_f, S_f) - CDF_t_Xb_f(bin_edges[:-1], x, y, C_f, D_f, S_f))
N_X_f_t_width = np.sqrt(N_X_f_t_mean)
N_Xb_f_t_width = np.sqrt(N_Xb_f_t_mean)
N_X_f_t_pull = (N_X_f_t-N_X_f_t_mean)/N_X_f_t_width
N_Xb_f_t_pull = (N_Xb_f_t-N_Xb_f_t_mean)/N_Xb_f_t_width
A_CP_t_width = (2*N_X_f_t_mean*N_Xb_f_t_mean)/(N_X_f_t_mean+N_Xb_f_t_mean)**2 * np.sqrt((N_X_f_t_width/N_X_f_t_mean)**2+(N_Xb_f_t_width/N_Xb_f_t_mean)**2)
A_CP_t_pull = (A_CP_t - A_CP(bin_midpoints, x, y, C_f, D_f, S_f))/A_CP_t_width


#Plot decay time generation of X->f
fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)
ax1 = fig.add_subplot(gs[0])
ax3 = fig.add_subplot(gs[1], sharex=ax1)

ax2 = ax1.twinx()
#ax1.set_title(r"Decay Time Distribution of $B^{0} \rightarrow J/\psi K^{0}_{S}$")
ax1.tick_params(labelbottom=False)
ax1.set_ylabel("Number of Events")
ax2.set_ylabel("Probability Density")
#ax1.hist(sample_t[sample_q == 1], bins=bin_edges, label=fr"Simulation, {len(sample_t[sample_q == 1])} Events")
#ax1.errorbar(bin_midpoints, N_X_f_t_mean, yerr=N_X_f_t_width, fmt="o", markersize=4, capsize=2, color="black", label="Expectation")
#ax2.plot(t_plot_range, PDF_t_X_f(t_plot_range, x, y, C_f, D_f, S_f), label=r"$\mathrm{PDF}(\Gamma t)$", color="red")
ax1.errorbar(bin_midpoints, N_X_f_t, yerr=np.sqrt(N_X_f_t), fmt="o", markersize=4, capsize=2, color="black", label="Simulation")
ax2.plot(t_plot_range, PDF_t_X_f(t_plot_range, x, y, C_f, D_f, S_f), color="blue", label=r"$\mathrm{PDF}(\Gamma t)$")
ax2.set_xlim(0, 5)
ax1.set_ylim(0, 1.1*N_X_f_t_mean[0])
ax2.set_ylim(0, 1.1*PDF_t_X_f(bin_midpoints[0], x, y, C_f, D_f, S_f))
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

ax3.errorbar(bin_midpoints, N_X_f_t_pull, yerr=1, fmt="o", markersize=4, capsize=2, color="black")
ax3.plot(np.array([t_plot_range[0], t_plot_range[-1]]), np.zeros(2), color="black", alpha=0.5)
ax3.set_xlim(0, 5)
ax3.set_ylim(-4, 4)
ax3.set_yticks([-2, 0, 2])
ax3.minorticks_off()
ax3.set_xlabel(r"Decay Time $\Gamma t$")
ax3.set_ylabel("Pull")

plt.savefig("decay_time_generation.pdf")
plt.show()


#Plot the CP asymmetry of the sample
fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

#ax1.set_title(r"Time-Dependent CP Asymmetry of the $B^{0}$-$\bar{B}^{0}$ System")
ax1.tick_params(labelbottom=False)
ax1.set_ylabel("CP Asymmetry")
ax1.errorbar(bin_midpoints, A_CP_t, yerr=A_CP_t_width, label="Simulation", fmt="o", markersize=4, capsize=2, color="black")
ax1.plot(t_plot_range, A_CP(t_plot_range, x, y, C_f, D_f, S_f), label="Expectation", color="blue")
ax1.plot(np.array([t_plot_range[0], t_plot_range[-1]]), np.zeros(2), color="black", linestyle="dashed")
ax1.set_xlim(0, 5)
ax1.legend(loc="upper right")

ax2.errorbar(bin_midpoints, A_CP_t_pull, yerr=1, fmt="o", markersize=4, capsize=2, color="black")
ax2.plot(np.array([t_plot_range[0], t_plot_range[-1]]), np.zeros(2), color="black", alpha=0.5)
ax2.set_xlim(0, 5)
ax2.set_ylim(-4, 4)
ax2.set_yticks([-2, 0, 2])
ax2.minorticks_off()
ax2.set_xlabel(r"Decay Time $\Gamma t$")
ax2.set_ylabel("Pull")

plt.savefig("time_dependent_CP_asymmetry.pdf")
plt.show()





#Define configuration
modulus_q_p = 1.0010
x = 0.7697
y = - 0.0005
C_f = 0.002
S_f = 0.711
D_f = - np.sqrt(1 - C_f**2 - S_f**2)

m_sig_mean = 5280
m_sig_width = 30
m_X_f_mean = m_sig_mean
m_X_f_width = m_sig_width
m_Xb_f_mean = m_sig_mean
m_Xb_f_width = m_sig_width
tau_bkg = 1
m_bkg_mean = 5280
m_bkg_width = 200

N_sig = 10**5
f_sig = 0.35
omega = 0.2
eps_X_f = 0.46
eps_Xb_f = 0.46

#Sample generation and fitting
sample_t, sample_m, sample_q = sample_generation_simple()
mi = Minuit(cost_function_simple, C_f=C_f, D_f=D_f, S_f=S_f, f_sig=f_sig)
mi.limits["C_f", "S_f"] = (-1, 1)
mi.fixed["D_f"] = True
mi.limits["f_sig"] = (0, 1)
mi.migrad()
mi.hesse()





#Store the iminuit results
C_f_value = mi.values["C_f"]
D_f_value = mi.values["D_f"]
S_f_value = mi.values["S_f"]
C_f_error = mi.errors["C_f"]
D_f_error = mi.errors["D_f"]
S_f_error = mi.errors["S_f"]
f_sig_value = mi.values["f_sig"]
f_sig_error = mi.errors["f_sig"]

#Compare the iminuit values to the true values
#Print true values
print("TRUE VALUE")
print(f"C_f = {C_f}")
print(f"D_f = {D_f}")
print(f"S_f = {S_f} \n")
#Print iminuit values
print("IMINUIT VALUE")
print(f"C_f = {C_f_value:.5f} \u00B1 {C_f_error:.5f}")
print(f"D_f = {D_f_value:.5f} \u00B1 {D_f_error:.5f}")
print(f"S_f = {S_f_value:.5f} \u00B1 {S_f_error:.5f}")