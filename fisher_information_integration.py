#Import packages
import numpy as np
import scipy





#Note that a constant acceptance function is assumed throughout this file
#The "_simple" suffix is not applied





#Define mixing and decay parameters
modulus_q_p = 1.0010
x = 0.7697
y = - 0.0005
C_f = 0.002
S_f = 0.711
D_f = - np.sqrt(1 - C_f**2 - S_f**2)

#Define other decay parameters
m_sig_mean = 5279.66
m_sig_width = 4
m_X_f_mean = m_sig_mean
m_X_f_width = m_sig_width
m_Xb_f_mean = m_sig_mean
m_Xb_f_width = m_sig_width
tau_bkg = 1
m_bkg_mean = 5279.66
m_bkg_width = 400

#Define detector, reconstruction, and selection parameters
#If f_sig=1, sample has no background events, recommend defining bkg parameters as np.nan
N_sig = 27 * 10**6
f_sig = 0.351
omega = 0.25
omega_array_cfg = np.linspace(0, 0.4, 17)
eps_sig = 0.470
eps_X_f = eps_sig
eps_Xb_f = eps_sig





#Define the PDF (divided by \Gamma) of  (\Gamma times) decay time
def PDF_t_X_f(t, x, y, C_f, D_f, S_f):
    factor_X_f = 1/(1-y**2) * (1+y*D_f)
    factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
    PDF_X_f = 1/(factor_X_f + factor_Y_f) * np.exp(-t) * (np.cosh(y*t)+D_f*np.sinh(y*t)+C_f*np.cos(x*t)-S_f*np.sin(x*t))
    return np.where(t>=0, PDF_X_f, 0)

def PDF_t_Xb_f(t, x, y, C_f, D_f, S_f):
    factor_X_f = 1/(1-y**2) * (1+y*D_f)
    factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
    PDF_Xb_f = 1/(factor_X_f - factor_Y_f) * np.exp(-t) * (np.cosh(y*t)+D_f*np.sinh(y*t)-C_f*np.cos(x*t)+S_f*np.sin(x*t))
    return np.where(t>=0, PDF_Xb_f, 0)

def PDF_t_bkg(t, tau_bkg):
    PDF_t_bkg = 1/tau_bkg * np.exp(-t/tau_bkg)
    return np.where(t>=0, PDF_t_bkg, 0)


#Define the PDF of invariant mass
def PDF_m_X_f(m, m_X_f_mean, m_X_f_width):
    PDF_m_X_f = 1/(np.sqrt(2*np.pi)*m_X_f_width) * np.exp(-((m-m_X_f_mean)**2)/(2*m_X_f_width**2))
    return np.where(m>=0, PDF_m_X_f, 0)

def PDF_m_Xb_f(m, m_Xb_f_mean, m_Xb_f_width):
    PDF_m_Xb_f = 1/(np.sqrt(2*np.pi)*m_Xb_f_width) * np.exp(-((m-m_Xb_f_mean)**2)/(2*m_Xb_f_width**2))
    return np.where(m>=0, PDF_m_Xb_f, 0)

def PDF_m_bkg(m, m_bkg_mean, m_bkg_width):
    PDF_m_bkg = 10**(-32)
    if (m>=(m_bkg_mean-0.5*m_bkg_width)) & (m<=(m_bkg_mean+0.5*m_bkg_width)):
        PDF_m_bkg = 1/m_bkg_width
    return np.where(m>=0, PDF_m_bkg, 0)





#Define the effective PDF, with simplification under the assumption of constant acceptance
def PDF(t, m, q):
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
    P_m_X_f = PDF_m_X_f(m, m_sig_mean, m_sig_width)
    P_m_Xb_f = PDF_m_Xb_f(m, m_sig_mean, m_sig_width)
    P_t_bkg = PDF_t_bkg(t, tau_bkg)
    P_m_bkg = PDF_m_bkg(m, m_bkg_mean, m_bkg_width)
    #Compute the overall effective PDF
    PDF_X_f = P_t_X_f*P_m_X_f*P_X_f_q_X_f*eps_X_f*P_X_f + P_t_Xb_f*P_m_Xb_f*P_Xb_f_q_X_f*eps_Xb_f*P_Xb_f + P_t_bkg*P_m_bkg*P_bkg_q_X_f*P_bkg
    PDF_Xb_f = P_t_X_f*P_m_X_f*P_X_f_q_Xb_f*eps_X_f*P_X_f + P_t_Xb_f*P_m_Xb_f*P_Xb_f_q_Xb_f*eps_Xb_f*P_Xb_f + P_t_bkg*P_m_bkg*P_bkg_q_Xb_f*P_bkg
    PDF = PDF_X_f*delta_q_X_f + PDF_Xb_f*delta_q_Xb_f
    return PDF





#Define the derivative of the PDF with respect to S_f
def PDF_der_S_f(t, m, q):
    #Define Kronecker delta for tag selection
    delta_q_X_f = np.where(q == 1, 1, 0)
    delta_q_Xb_f = np.where(q == -1, 1, 0)
    #Compute ratio of branching fractions
    factor_X_f = 1/(1-y**2) * (1+y*D_f)
    factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
    BF_ratio_Xb_f_X_f = 1/(modulus_q_p**2) * (factor_X_f - factor_Y_f)/(factor_X_f + factor_Y_f)
    #Compute the derivative of the effective PDF with respect to S_f, with the assumption that modulus(q/p) = 1
    PDF_der_S_f_q_X_f = 1/(2*factor_X_f) * np.exp(-t) * (1 - 2*omega)*(- np.sin(x*t)) * PDF_m_X_f(m, m_sig_mean, m_sig_width) * f_sig
    PDF_der_S_f_q_Xb_f = 1/(2*factor_X_f) * np.exp(-t) * (-1)*(1 - 2*omega)*(- np.sin(x*t)) * PDF_m_X_f(m, m_sig_mean, m_sig_width) * f_sig
    PDF_der_S_f = PDF_der_S_f_q_X_f*delta_q_X_f + PDF_der_S_f_q_Xb_f*delta_q_Xb_f
    return PDF_der_S_f





#Define the integrand of the Fisher information matrxi element for S_f
#Summation over the tag q is performed on the integrand
#Note that the order of integration needs to be swapped in accordance with the convention of scipy.integrate.dblquad
def integrand_q_summed(m, t):
    integrand_q_summed = (PDF_der_S_f(t, m, q=1)**2)/PDF(t, m, q=1) + (PDF_der_S_f(t, m, q=-1)**2)/PDF(t, m, q=-1)
    return integrand_q_summed





#Simplifications made:
#modulus_q_p = 1, such that the factor_Y_f (and hence C_f and S_f) dependence vanishes from the product of the decay time PDF normalisation and the ratio of the time-integrated branching fractions
#eps_avg_X_f = eps_avg_Xb_f, or the stronger condition eps_X_f = eps_Xb_f = constant, to remove dependence of the raiot of the time-integrated branching fractions from the normalisation
#eps_avg_X_f = eps_avg_Xb_f also simplifies the expression of N_acc, but this is not compulsory
#The diagonal elements of the inverse Fisher information matrix is approximated as the inverse of the diagonal elements of the matrix
#Warning: the Fisher information error returned later only reflects the error in scipy integration, any error due to the matrix inversion assumption is not accounted for





#Initialise fisher information arrays
per_event_fisher_information_S_f_value_array_cfg = np.zeros(len(omega_array_cfg))
per_event_fisher_information_S_f_error_array_cfg = np.zeros(len(omega_array_cfg))

#Compute the per-event fisher information with respect to S_f
for cfg_index in range(len(omega_array_cfg)):
    omega = omega_array_cfg[cfg_index]
    per_event_fisher_information_S_f_value_array_cfg[cfg_index], per_event_fisher_information_S_f_error_array_cfg[cfg_index] = scipy.integrate.dblquad(integrand_q_summed, 0, 100, 5000, 5500)

#Save the results
np.save("per_event_fisher_information_S_f_value_array_cfg.npy", per_event_fisher_information_S_f_value_array_cfg)
np.save("per_event_fisher_information_S_f_error_array_cfg.npy", per_event_fisher_information_S_f_error_array_cfg)