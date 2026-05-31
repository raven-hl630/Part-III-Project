#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numba_stats import norm, expon, uniform
import scipy
import matplotlib.pyplot as plt
#import mplhep as hep
import iminuit
from iminuit import Minuit, cost
import time


# In[3]:


#List of detector variables
#Note that these variables should be re-defined in the fitting section

#Expected number of produced signal events from FCC
#Source: PDG
N_Z = 3 * 10**12
BF_Z_bb = 0.1512
f_b_B = 0.408
BF_B_Jpsi_KS = 0.5 * 8.91 * 10**(-4)
BF_Jpsi_e_e = 0.05971
BF_Jpsi_mu_mu = 0.05961
BF_KS_pi_pi = 0.6920
epsilon_sig = 0.47

N_sig_expected = N_Z * BF_Z_bb * 2 * f_b_B * BF_B_Jpsi_KS * (BF_Jpsi_e_e + BF_Jpsi_mu_mu) * BF_KS_pi_pi
print(f"N_sig_expected = {N_sig_expected}")


#Expected signal fraction
p_sig_expected = 14/(14+36)
print(f"p_sig_expected = {p_sig_expected}")

#Expected signal acceptance
eps_sig_expected = 0.46
print(f"eps_sig_expected = {eps_sig_expected}")


# In[3]:


#List of CP violation and decay variables
#Note that these variables should be re-defined in the fitting section

#Define the mixing parameters
#Source: HFLAV
modulus_q_p = 1.0010
x = 0.7697
y = - 0.0005     #where modulus of y < 1 for convergence

#Define the CP violation parameters
#Source: PDG
C_f = 0.002
S_f = 0.711                                #S = sin(2 beta), 2 beta = 43 degrees
D_f = - np.sqrt(1 - C_f**2 - S_f**2)       #D = - cos(2 beta); positive value of D_f also possible in theory

#Giving the lambda_f parameter as
modulus_lambda_f = np.sqrt((1-C_f)/(1+C_f))
phase_lambda_f = np.arctan2(S_f, D_f)

#Compute the ratio of the branching fractions B(Xb->f) to B(X->f)
factor_X_f = 1/(1-y**2) * (1+y*D_f)
factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
BF_ratio_Xb_f_X_f = 1/(modulus_q_p**2) * (factor_X_f - factor_Y_f)/(factor_X_f + factor_Y_f)

print(f"C_f = {C_f}")
print(f"D_f = {D_f}")
print(f"S_f = {S_f}")
print(f"B(Xb->f)/B(X->f) = {BF_ratio_Xb_f_X_f}")
print(f"modulus_lambda_f = {modulus_lambda_f}")
print(f"phase_lambda_f = {phase_lambda_f}")


# In[4]:


#Define PDFs for decay time and invariant mass

#Define (PDF / Gamma) of (Gamma * t) decay time
#Note that all normalisation factors are functions of the CDS parameters, which are to be fitted, hence must be computed locally within the PDFs
def PDF_t_X_f(t, C_f, D_f, S_f):
    factor_X_f = 1/(1-y**2) * (1+y*D_f)
    factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
    PDF_X_f = 1/(factor_X_f + factor_Y_f) * np.exp(-t) * (np.cosh(y*t)+D_f*np.sinh(y*t)+C_f*np.cos(x*t)-S_f*np.sin(x*t))
    return np.where(t>=0, PDF_X_f, 0)

def PDF_t_Xb_f(t, C_f, D_f, S_f):
    factor_X_f = 1/(1-y**2) * (1+y*D_f)
    factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
    PDF_Xb_f = 1/(factor_X_f - factor_Y_f) * np.exp(-t) * (np.cosh(y*t)+D_f*np.sinh(y*t)-C_f*np.cos(x*t)+S_f*np.sin(x*t))
    return np.where(t>=0, PDF_Xb_f, 0)

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


# In[5]:


#Define CDFs for decay time and invariant mass

#Define the CDF of (Gamma * decay time) for use in event generation and fitting
def CDF_t_X_f(t, C_f, D_f, S_f):
    factor_X_f = 1/(1-y**2) * (1+y*D_f)
    factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
    CDF_X_f = - 1/(factor_X_f + factor_Y_f) * ( np.exp(-t) * (1/(1-y**2)*((1+y*D_f)*np.cosh(y*t)+(y+D_f)*np.sinh(y*t)) + 1/(1+x**2)*((C_f-x*S_f)*np.cos(x*t)+(-x*C_f-S_f)*np.sin(x*t))) - (1/(1-y**2)*(1+y*D_f) + 1/(1+x**2)*(C_f-x*S_f)) )
    return np.where(t>=0, CDF_X_f, 0)

def CDF_t_Xb_f(t, C_f, D_f, S_f):
    factor_X_f = 1/(1-y**2) * (1+y*D_f)
    factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
    CDF_Xb_f = - 1/(factor_X_f - factor_Y_f) * ( np.exp(-t) * (1/(1-y**2)*((1+y*D_f)*np.cosh(y*t)+(y+D_f)*np.sinh(y*t)) + 1/(1+x**2)*((-C_f+x*S_f)*np.cos(x*t)+(x*C_f+S_f)*np.sin(x*t))) - (1/(1-y**2)*(1+y*D_f) + 1/(1+x**2)*(-C_f+x*S_f)) )
    return np.where(t>=0, CDF_Xb_f, 0)

def CDF_t_bkg(t, tau_bkg):
    CDF_t_bkg = 1 - np.exp(-t/tau_bkg)
    return np.where(t>=0, CDF_t_bkg, 0)

#Define the CDF of invariant mass for use in event generation and fitting
def CDF_m_X_f(m, m_X_f_mean, m_X_f_width):
    CDF_m_X_f = np.where(m>=0, scipy.integrate.quad(lambda m: PDF_m_X_f(m, m_X_f_mean, m_X_f_width), 0, m), 0)
    return CDF_m_X

def CDF_m_Xb_f(m, m_Xb_f_mean, m_Xb_f_width):
    CDF_m_Xb_f = np.where(m>=0, scipy.integrate.quad(lambda m: PDF_m_Xb_f(m, m_Xb_f_mean, m_Xb_f_width), 0, m), 0)
    return CDF_m_Xb_f

def CDF_m_bkg(m, m_bkg_mean, m_bkg_width):
    CDF_m_bkg = np.zeros(len(m))
    CDF_m_bkg[(m<(m_bkg_mean-0.5*m_bkg_width))] = 0
    CDF_m_bkg[(m>=(m_bkg_mean-0.5*m_bkg_width)) & (m<=(m_bkg_mean+0.5*m_bkg_width))] = 1/m_bkg_width * (m - m_bkg_mean) + 0.5
    CDF_m_bkg[(m>(m_bkg_mean+0.5*m_bkg_width))] = 1
    return CDF_m_bkg


# In[6]:


#Define the CP asymmetry
def A_CP(t, C_f, D_f, S_f):
    factor_X_f = 1/(1-y**2) * (1+y*D_f)
    factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
    BF_ratio_Xb_f_X_f = 1/(modulus_q_p**2) * (factor_X_f - factor_Y_f)/(factor_X_f + factor_Y_f)
    factor_U_div_B_X_f = 2/(factor_X_f + factor_Y_f)
    PDF_t_X_f = factor_U_div_B_X_f * (1/2) * np.exp(-t) * (np.cosh(y*t)+D_f*np.sinh(y*t)+C_f*np.cos(x*t)-S_f*np.sin(x*t))
    factor_U_div_B_Xb_f = 2/(factor_X_f - factor_Y_f) * (modulus_q_p**2)
    PDF_t_Xb_f = factor_U_div_B_Xb_f * 1/(modulus_q_p**2) * (1/2) * np.exp(-t) * (np.cosh(y*t)+D_f*np.sinh(y*t)-C_f*np.cos(x*t)+S_f*np.sin(x*t))
    A_CP = (- PDF_t_X_f + BF_ratio_Xb_f_X_f*PDF_t_Xb_f)/(PDF_t_X_f + BF_ratio_Xb_f_X_f*PDF_t_Xb_f)
    return A_CP


# In[7]:


#Define sample generation and fitting functions for general cases

#Define the sampling function that returns samples of decay time, invariant mass, and tag
#C_f, D_f, S_f, m_X_f_mean, m_X_f_width, m_Xb_f_mean, m_Xb_f_width, tau_bkg, m_bkg_mean, m_bkg_width, N_sig, p_sig, eps_X_f, eps_Xb_f, eps_bkg, omega
def sample_generation():

    #Compute the ratio of branching fractions
    factor_X_f = 1/(1-y**2) * (1+y*D_f)
    factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
    BF_ratio_Xb_f_X_f = 1/(modulus_q_p**2) * (factor_X_f - factor_Y_f)/(factor_X_f + factor_Y_f)
    #Compute numbers of true X->f signal events, X->f signal events, and background events
    N_X_f = np.random.poisson(1/(1+BF_ratio_Xb_f_X_f) * N_sig)
    N_Xb_f = np.random.poisson(BF_ratio_Xb_f_X_f/(1+BF_ratio_Xb_f_X_f) * N_sig)
    N_bkg = np.random.poisson((1-p_sig)/p_sig * N_sig)

    #Generate decay time data for signal and background events
    #Using a sample from a uniform distribution between 0 and 1 as the argument of the inverse CDF gives a sample from the corresponding probability distribution
    #For signal events
    u_t = np.random.uniform(0, 1, N_X_f)
    v_t = np.random.uniform(0, 1, N_Xb_f)
    t_range = np.linspace(0, 100, 100000)
    sample_t_X_f = np.interp(u_t, CDF_t_X_f(t_range, C_f, D_f, S_f), t_range)
    sample_t_Xb_f = np.interp(v_t, CDF_t_Xb_f(t_range, C_f, D_f, S_f), t_range)
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

    #Incorporate acceptance
    #Assume an X/Xb->f/bkg event occuring at decay time t has a probability eps_X_f/Xb_f/bkg(t) to be accepted
    u_acc = np.random.uniform(0, 1, N_X_f)
    v_acc = np.random.uniform(0, 1, N_Xb_f)
    w_acc = np.random.uniform(0, 1, N_bkg)
    #If u/v/w <= eps(t), the event is accepted; if u/v/w > eps(t), the event is not accepted (if eps=1, all events are accepted)
    #Compute the acceptance
    acceptance_X_f = (u_acc <= eps_X_f(sample_t_X_f))
    acceptance_Xb_f = (v_acc <= eps_Xb_f(sample_t_Xb_f))
    acceptance_bkg = (w_acc <= eps_bkg(sample_t_bkg))
    #Retain only the acceptanced events
    sample_acc_t_X_f = sample_t_X_f[acceptance_X_f]
    sample_acc_t_Xb_f = sample_t_Xb_f[acceptance_Xb_f]
    sample_acc_t_bkg = sample_t_bkg[acceptance_bkg]
    sample_acc_m_X_f = sample_m_X_f[acceptance_X_f]
    sample_acc_m_Xb_f = sample_m_Xb_f[acceptance_Xb_f]
    sample_acc_m_bkg = sample_m_bkg[acceptance_bkg]
    N_acc_X_f = np.sum(acceptance_X_f)
    N_acc_Xb_f = np.sum(acceptance_Xb_f)
    N_acc_bkg = np.sum(acceptance_bkg)

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
    sample_t = np.concatenate((sample_acc_tag_t_X_f, sample_acc_tag_t_Xb_f, sample_acc_t_bkg))
    sample_m = np.concatenate((sample_acc_tag_m_X_f, sample_acc_tag_m_Xb_f, sample_acc_m_bkg))
    sample_q = np.concatenate((np.ones(N_acc_tag_X_f), -np.ones(N_acc_tag_Xb_f), np.random.choice([1, -1], size=N_acc_bkg, p=[0.5, 0.5])))

    return sample_t, sample_m, sample_q


#Define the PDF of a dataset (t, m, q)
#Note that the PDF function should explicitly take both the dataset (t, m, q) and the fitted parameter set (C_f, D_f, S_f, p_sig) as arguments
def PDF(t, m, q, C_f, D_f, S_f, p_sig):
    #Define Kronecker delta for tag selection
    delta_q_X_f = np.where(q == 1, 1, 0)
    delta_q_Xb_f = np.where(q == -1, 1, 0)
    #Compute ratio of branching fractions
    factor_X_f = 1/(1-y**2) * (1+y*D_f)
    factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
    BF_ratio_Xb_f_X_f = 1/(modulus_q_p**2) * (factor_X_f - factor_Y_f)/(factor_X_f + factor_Y_f)
    #Compute true decay source probabilities P(X/Xb/bkg)
    P_X_f = p_sig * 1/(1+BF_ratio_Xb_f_X_f)
    P_Xb_f = p_sig * BF_ratio_Xb_f_X_f/(1+BF_ratio_Xb_f_X_f)
    P_bkg = 1 - p_sig
    #Compute the acceptance
    P_eps_X_f = eps_X_f(t)
    P_eps_Xb_f = eps_Xb_f(t)
    P_eps_bkg = eps_bkg(t)
    #Compute average acceptance
    eps_avg_X_f, eps_avg_error_X_f = scipy.integrate.quad(lambda t: PDF_t_X_f(t, C_f, D_f, S_f)*eps_X_f(t), 0, np.inf)
    eps_avg_Xb_f, eps_avg_error_Xb_f = scipy.integrate.quad(lambda t: PDF_t_Xb_f(t, C_f, D_f, S_f)*eps_Xb_f(t), 0, np.inf)
    eps_avg_bkg, eps_avg_error_bkg = scipy.integrate.quad(lambda t: PDF_t_bkg(t, tau_bkg)*eps_bkg(t), 0, np.inf)
    #Compute conditional tagging probabilities given the true source P(q=X/Xb|X/Xb/bkg)
    #Define P_A_q_B as the probability of A being tagged as q=B
    P_X_f_q_X_f = 1 - omega
    P_X_f_q_Xb_f = omega
    P_Xb_f_q_X_f = omega
    P_Xb_f_q_Xb_f = 1 - omega
    P_bkg_q_X_f = 0.5
    P_bkg_q_Xb_f = 0.5
    #Compute sample PDFs of decay time and invariant mass given the true source
    P_t_X_f = PDF_t_X_f(t, C_f, D_f, S_f)
    P_t_Xb_f = PDF_t_Xb_f(t, C_f, D_f, S_f)
    P_m_X_f = PDF_m_X_f(m, m_X_f_mean, m_X_f_width)
    P_m_Xb_f = PDF_m_Xb_f(m, m_Xb_f_mean, m_Xb_f_width)
    P_t_bkg = PDF_t_bkg(t, tau_bkg)
    P_m_bkg = PDF_m_bkg(m, m_bkg_mean, m_bkg_width)
    #P_m_X_f = norm.pdf(m, m_X_f_mean, m_X_f_width)
    #P_m_Xb_f = norm.pdf(m, m_Xb_f_mean, m_Xb_f_width)
    #P_t_bkg = expon.pdf(t, 0, tau_bkg)
    #P_m_bkg = uniform.pdf(m, m_bkg_mean - 0.5*m_bkg_width, m_bkg_width)
    #Compute the overall effective PDF
    PDF_X_f = P_t_X_f*P_m_X_f*P_X_f_q_X_f*P_eps_X_f*P_X_f + P_t_Xb_f*P_m_Xb_f*P_Xb_f_q_X_f*P_eps_Xb_f*P_Xb_f + P_t_bkg*P_m_bkg*P_bkg_q_X_f*P_eps_bkg*P_bkg
    PDF_Xb_f = P_t_X_f*P_m_X_f*P_X_f_q_Xb_f*P_eps_X_f*P_X_f + P_t_Xb_f*P_m_Xb_f*P_Xb_f_q_Xb_f*P_eps_Xb_f*P_Xb_f + P_t_bkg*P_m_bkg*P_bkg_q_Xb_f*P_eps_bkg*P_bkg
    normalisation = p_sig*(eps_avg_X_f*1/(1+BF_ratio_Xb_f_X_f) + eps_avg_Xb_f*BF_ratio_Xb_f_X_f/(1+BF_ratio_Xb_f_X_f)) + (1-p_sig)*eps_avg_bkg
    PDF = 1/normalisation * (PDF_X_f*delta_q_X_f + PDF_Xb_f*delta_q_Xb_f)
    return PDF

#Define the unbinned NLL cost function
#Note that iminuit fits all the arguments of the cost function; sample data and parameters not meant to be fitted must be read globally
def cost_function(C_f, D_f, S_f, p_sig):
    #PDF_sample = np.clip(PDF(sample_t, sample_m, sample_q, C_f, D_f, S_f, p_sig), 1e-300, None)
    PDF_sample = PDF(sample_t, sample_m, sample_q, C_f, D_f, S_f, p_sig)
    cost_function = - np.sum(np.log(PDF_sample))
    return cost_function
#Inform iminuit the cost function is in the NLL form
cost_function.errordef = Minuit.LIKELIHOOD


# In[8]:


#Define sample generation and fitting functions, assuming constant acceptance functions

#t_range = np.linspace(0, 100, 100000)
#CDF_t_X_f_grid = CDF_t_X_f(t_range, C_f, D_f, S_f)
#CDF_t_Xb_f_grid = CDF_t_Xb_f(t_range, C_f, D_f, S_f)
#def inversion(u, CDF_grid):
#    index = np.searchsorted(CDF_grid, u, side="right")
#    t0 = t_range[index - 1]
#    t1 = t_range[index]
#    f0 = CDF_grid[index-1]
#    f1 = CDF_grid[index]
#    t_inverted = t0 + (u - f0)/(f1 - f0) * (t1 - t0)
#    return t_inverted
#    sample_t_X_f = inversion(u_t, CDF_t_X_f_grid)
#    sample_t_Xb_f = inversion(v_t, CDF_t_Xb_f_grid)

#Define the sampling function that returns samples of decay time, invariant mass, and tag
#C_f, D_f, S_f, m_X_f_mean, m_X_f_width, m_Xb_f_mean, m_Xb_f_width, tau_bkg, m_bkg_mean, m_bkg_width, N_sig, p_sig, eps_X_f, eps_Xb_f, eps_bkg, omega
def sample_generation_simple():

    #Compute the ratio of branching fractions
    factor_X_f = 1/(1-y**2) * (1+y*D_f)
    factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
    BF_ratio_Xb_f_X_f = 1/(modulus_q_p**2) * (factor_X_f - factor_Y_f)/(factor_X_f + factor_Y_f)
    #Compute numbers of true X->f signal events, X->f signal events, and background events
    N_X_f = np.random.poisson(1/(1+BF_ratio_Xb_f_X_f) * N_sig)
    N_Xb_f = np.random.poisson(BF_ratio_Xb_f_X_f/(1+BF_ratio_Xb_f_X_f) * N_sig)
    N_bkg = np.random.poisson((1-p_sig)/p_sig * N_sig)

    #Generate decay time data for signal and background events
    #Using a sample from a uniform distribution between 0 and 1 as the argument of the inverse CDF gives a sample from the corresponding probability distribution
    #For signal events
    u_t = np.random.uniform(0, 1, N_X_f)
    v_t = np.random.uniform(0, 1, N_Xb_f)
    t_range = np.linspace(0, 100, 100000)
    sample_t_X_f = np.interp(u_t, CDF_t_X_f(t_range, C_f, D_f, S_f), t_range)
    sample_t_Xb_f = np.interp(v_t, CDF_t_Xb_f(t_range, C_f, D_f, S_f), t_range)
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

    #Incorporate acceptance
    #Assume an X/Xb->f/bkg event occuring at decay time t has a probability eps_X_f/Xb_f/bkg(t) to be accepted
    u_acc = np.random.uniform(0, 1, N_X_f)
    v_acc = np.random.uniform(0, 1, N_Xb_f)
    w_acc = np.random.uniform(0, 1, N_bkg)
    #If u/v/w <= eps(t), the event is accepted; if u/v/w > eps(t), the event is not accepted (if eps=1, all events are accepted)
    #Compute the acceptance
    acceptance_X_f = (u_acc <= eps_X_f)
    acceptance_Xb_f = (v_acc <= eps_Xb_f)
    acceptance_bkg = (w_acc <= eps_bkg)
    #Retain only the acceptanced events
    sample_acc_t_X_f = sample_t_X_f[acceptance_X_f]
    sample_acc_t_Xb_f = sample_t_Xb_f[acceptance_Xb_f]
    sample_acc_t_bkg = sample_t_bkg[acceptance_bkg]
    sample_acc_m_X_f = sample_m_X_f[acceptance_X_f]
    sample_acc_m_Xb_f = sample_m_Xb_f[acceptance_Xb_f]
    sample_acc_m_bkg = sample_m_bkg[acceptance_bkg]
    N_acc_X_f = np.sum(acceptance_X_f)
    N_acc_Xb_f = np.sum(acceptance_Xb_f)
    N_acc_bkg = np.sum(acceptance_bkg)

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
    sample_t = np.concatenate((sample_acc_tag_t_X_f, sample_acc_tag_t_Xb_f, sample_acc_t_bkg))
    sample_m = np.concatenate((sample_acc_tag_m_X_f, sample_acc_tag_m_Xb_f, sample_acc_m_bkg))
    sample_q = np.concatenate((np.ones(N_acc_tag_X_f), -np.ones(N_acc_tag_Xb_f), np.random.choice([1, -1], size=N_acc_bkg, p=[0.5, 0.5])))

    return sample_t, sample_m, sample_q


#Define the effective PDF of a dataset (t, m, q)
#Note that the PDF function should explicitly take both the dataset (t, m, q) and the fitted parameter set (C_f, D_f, S_f, p_sig) as arguments
def PDF_simple(t, m, q, C_f, D_f, S_f, p_sig):
    #Define Kronecker delta for tag selection
    delta_q_X_f = np.where(q == 1, 1, 0)
    delta_q_Xb_f = np.where(q == -1, 1, 0)
    #Compute ratio of branching fractions
    factor_X_f = 1/(1-y**2) * (1+y*D_f)
    factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
    BF_ratio_Xb_f_X_f = 1/(modulus_q_p**2) * (factor_X_f - factor_Y_f)/(factor_X_f + factor_Y_f)
    #Compute true decay source probabilities P(X/Xb/bkg)
    P_X_f = p_sig * 1/(1+BF_ratio_Xb_f_X_f)
    P_Xb_f = p_sig * BF_ratio_Xb_f_X_f/(1+BF_ratio_Xb_f_X_f)
    P_bkg = 1 - p_sig
    #Compute conditional tagging probabilities given the true source P(q=X/Xb|X/Xb/bkg)
    #Define P_A_q_B as the probability of A being tagged as q=B
    P_X_f_q_X_f = 1 - omega
    P_X_f_q_Xb_f = omega
    P_Xb_f_q_X_f = omega
    P_Xb_f_q_Xb_f = 1 - omega
    P_bkg_q_X_f = 0.5
    P_bkg_q_Xb_f = 0.5
    #Compute sample PDFs of decay time and invariant mass given the true source
    P_t_X_f = PDF_t_X_f(t, C_f, D_f, S_f)
    P_t_Xb_f = PDF_t_Xb_f(t, C_f, D_f, S_f)
    P_m_X_f = PDF_m_X_f(m, m_X_f_mean, m_X_f_width)
    P_m_Xb_f = PDF_m_Xb_f(m, m_Xb_f_mean, m_Xb_f_width)
    P_t_bkg = PDF_t_bkg(t, tau_bkg)
    P_m_bkg = PDF_m_bkg(m, m_bkg_mean, m_bkg_width)
    #P_m_X_f = norm.pdf(m, m_X_f_mean, m_X_f_width)
    #P_m_Xb_f = norm.pdf(m, m_Xb_f_mean, m_Xb_f_width)
    #P_t_bkg = expon.pdf(t, 0, tau_bkg)
    #P_m_bkg = uniform.pdf(m, m_bkg_mean - 0.5*m_bkg_width, m_bkg_width)
    #Compute the overall effective PDF
    PDF_X_f = P_t_X_f*P_m_X_f*P_X_f_q_X_f*eps_X_f*P_X_f + P_t_Xb_f*P_m_Xb_f*P_Xb_f_q_X_f*eps_Xb_f*P_Xb_f + P_t_bkg*P_m_bkg*P_bkg_q_X_f*eps_bkg*P_bkg
    PDF_Xb_f = P_t_X_f*P_m_X_f*P_X_f_q_Xb_f*eps_X_f*P_X_f + P_t_Xb_f*P_m_Xb_f*P_Xb_f_q_Xb_f*eps_Xb_f*P_Xb_f + P_t_bkg*P_m_bkg*P_bkg_q_Xb_f*eps_bkg*P_bkg
    normalisation = p_sig*(eps_X_f*1/(1+BF_ratio_Xb_f_X_f) + eps_Xb_f*BF_ratio_Xb_f_X_f/(1+BF_ratio_Xb_f_X_f)) + (1-p_sig)*eps_bkg
    PDF_simple = 1/normalisation * (PDF_X_f*delta_q_X_f + PDF_Xb_f*delta_q_Xb_f)
    return PDF_simple

#Define the unbinned NLL cost function
#Note that iminuit fits all the arguments of the cost function; sample data and parameters not meant to be fitted must be read globally
def cost_function_simple(C_f, D_f, S_f, p_sig):
    #PDF_sample_simple = np.clip(PDF_simple(sample_t, sample_m, sample_q, C_f, D_f, S_f, p_sig), 1e-300, None)
    PDF_sample_simple = PDF_simple(sample_t, sample_m, sample_q, C_f, D_f, S_f, p_sig)
    cost_function_simple = - np.sum(np.log(PDF_sample_simple))
    return cost_function_simple
#Inform iminuit the cost function is in the NLL form
cost_function_simple.errordef = Minuit.LIKELIHOOD


# In[9]:


#Check normalisation
print(f"Int PDF_t_X_f = {scipy.integrate.quad(lambda t: PDF_t_X_f(t, C_f, D_f, S_f), 0, np.inf)}")
print(f"Int PDF_t_Xb_f = {scipy.integrate.quad(lambda t: PDF_t_Xb_f(t, C_f, D_f, S_f), 0, np.inf)}")
print(f"CDF_t_X_f(100) = {CDF_t_X_f(100, C_f, D_f, S_f)}")
print(f"CDF_t_Xb_f(100) = {CDF_t_Xb_f(100, C_f, D_f, S_f)}")

#Plot PDFs, CDFs, and CP asymmetry
t_plot_range = np.linspace(0, 5, 1000)
fig, ax = plt.subplots()
ax.set_title(r"PDF of Decay Time for $X^{0}$ and $\bar{X}^{0}$")
ax.set_xlabel(r"Decay Time, $\Gamma t$")
ax.set_ylabel("Probability (Density)")
ax.plot(t_plot_range, PDF_t_X_f(t_plot_range, C_f, D_f, S_f), label=r"$X^{0} \rightarrow f \quad \mathrm{PDF}(\Gamma t) / \Gamma$", color="blue")
ax.plot(t_plot_range, CDF_t_X_f(t_plot_range, C_f, D_f, S_f), label=r"$X^{0} \rightarrow f \quad \mathrm{CDF}(\Gamma t)$", color="blue", linestyle=(0,(5,5)))
ax.plot(t_plot_range, PDF_t_Xb_f(t_plot_range, C_f, D_f, S_f), label=r"$\bar{X}^{0} \rightarrow f \quad \mathrm{PDF}(\Gamma t) / \Gamma$", color="red")
ax.plot(t_plot_range, CDF_t_Xb_f(t_plot_range, C_f, D_f, S_f), label=r"$\bar{X}^{0} \rightarrow f \quad \mathrm{CDF}(\Gamma t)$", color="red", linestyle=(5,(5,5)))
line_legend = ax.legend(loc="upper right")
parameter_legend = ax.legend([rf"$C_{{f}}$ = {C_f:.3f}", rf"$D_{{f}}$ = {D_f:.3f}", rf"$S_{{f}}$ = {S_f:.3f}"], handlelength=0, loc="upper center")
ax.add_artist(line_legend)
ax.grid(True)
plt.show()

#Plot PDFs, CDFs, and CP asymmetry
t_list = np.linspace(0, 5, 1000)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_title(r"PDF of Decay Time for $X^{0}$ and $\bar{X}^{0}$")
ax1.set_xlabel(r"Decay Time, $\Gamma t$")
ax1.set_ylabel(r"Probability Density / $\Gamma$")
ax2.set_ylabel("Cumulative Probability")
ax1.plot(t_plot_range, PDF_t_X_f(t_plot_range, C_f, D_f, S_f), label=r"$X^{0} \rightarrow f \quad \mathrm{PDF}(\Gamma t) / \Gamma$", color="blue")
ax1.plot(t_plot_range, PDF_t_Xb_f(t_plot_range, C_f, D_f, S_f), label=r"$\bar{X}^{0} \rightarrow f \quad \mathrm{PDF}(\Gamma t) / \Gamma$", color="red")
ax2.plot(t_plot_range, CDF_t_X_f(t_plot_range, C_f, D_f, S_f), label=r"$X^{0} \rightarrow f \quad \mathrm{CDF}(\Gamma t)$", color="blue", linestyle=(0,(5,5)))
ax2.plot(t_plot_range, CDF_t_Xb_f(t_plot_range, C_f, D_f, S_f), label=r"$\bar{X}^{0} \rightarrow f \quad \mathrm{CDF}(\Gamma t)$", color="red", linestyle=(5,(5,5)))
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
line_legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
parameter_legend = ax1.legend([rf"$C_{{f}}$ = {C_f:.3f}", rf"$D_{{f}}$ = {D_f:.3f}", rf"$S_{{f}}$ = {S_f:.3f}"], handlelength=0, loc="center")
ax1.add_artist(line_legend)
ax1.grid(True)
plt.show()


# In[10]:


#Define CP violation and decay parameters
modulus_q_p = 1.0010
x = 0.7697
y = - 0.0005
C_f = 0.002
S_f = 0.711
D_f = - np.sqrt(1 - C_f**2 - S_f**2)

#Define decay time and invariant mass parameters for signals and background (except for signal decay time parameters CDS, which are defined at the start)
#Here, p_sig=1 gives no background events, bkg parameters are defined as np.nan
m_sig_mean = 100
m_sig_width = 1
m_X_f_mean = m_sig_mean
m_X_f_width = m_sig_width
m_Xb_f_mean = m_sig_mean
m_Xb_f_width = m_sig_width
tau_bkg = np.nan
m_bkg_mean = np.nan
m_bkg_width = np.nan

#Define detector parameters
#N_sig is a run parameter
#p_sig is a run parameter
#omega is a run parameter
eps_X_f = 1
eps_Xb_f = 2
eps_bkg = 1

#Define run parameters
N_sig = 10**5
p_sig = 1
omega = 0


#Generate samples
sample_t, sample_m, sample_q = sample_generation_simple()
#Plot histograms of samples
t_plot_range = np.linspace(0, 5, 1000)
bin_edges = np.linspace(0, 5, 100)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_title(r"Decay Time Distribution of $X^{0} \rightarrow f$")
ax1.set_xlabel(r"Decay Time, $\Gamma t$")
ax1.set_ylabel("Number of Events")
ax2.set_ylabel(r"Probability Density / $\Gamma$")
ax1.hist(sample_t[sample_q == 1], bins=bin_edges, label=fr"$N = {len(sample_t[sample_q == 1])}$", color="#1f77b4")
ax2.plot(t_plot_range, PDF_t_X_f(t_list, C_f, D_f, S_f), label=r"$X^{0} \rightarrow f \quad \mathrm{PDF}(\Gamma t) / \Gamma$", color="red")
#ax2.plot(t_plot_range, CDF_t_X_f(t_list, C_f, D_f, S_f), label=r"$X^{0} \rightarrow f \quad \mathrm{CDF}(\Gamma t)$", color="black")
ax2.set_xlim(0, 5)
ax2.set_ylim(bottom=0)
ax1.legend(loc="upper center")
ax2.legend(loc="upper right")
plt.show()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_title(r"Decay Time Distribution of $\bar{X}^{0} \rightarrow f$")
ax1.set_xlabel(r"Decay Time, $\Gamma t$")
ax1.set_ylabel("Number of Events")
ax2.set_ylabel(r"Probability Density / $\Gamma$")
ax1.hist(sample_t[sample_q == -1], bins=bin_edges, label=fr"$N = {len(sample_t[sample_q == -1])}$", color="#1f77b4")
ax2.plot(t_plot_range, PDF_t_Xb_f(t_list, C_f, D_f, S_f), label=r"$\bar{X}^{0} \rightarrow f \quad \mathrm{PDF}(\Gamma t) / \Gamma$", color="red")
#ax2.plot(t_plot_range, CDF_t_Xb_f(t_list, C_f, D_f, S_f), label=r"$\bar{X}^{0} \rightarrow f \quad \mathrm{CDF}(\Gamma t)$", color="black")
ax2.set_xlim(0, 5)
ax2.set_ylim(bottom=0)
ax1.legend(loc="upper center")
ax2.legend(loc="upper right")
plt.show()

#Bin the samples
bin_edges = np.linspace(0, 5, 100)
bin_midpoints = 0.5*(bin_edges[:-1]+bin_edges[1:])
N_X_f_t, edge_X_f = np.histogram(sample_t[sample_q == 1], bins=bin_edges)
N_Xb_f_t, edge_Xb_f = np.histogram(sample_t[sample_q == -1], bins=bin_edges)
A_CP_t = (N_Xb_f_t-N_X_f_t)/(N_X_f_t+N_Xb_f_t)

#Define errors in N_X_f_t and N_Xb_f_t
expectation_N_X_f_t = N_X_f_t       #expectation_N_X_f_t = 0.5*N*(CDF_X_f(bin_edges[1:])-CDF_X_f(bin_edges[:-1]))
expectation_N_Xb_f_t = N_Xb_f_t     #expectation_N_Xb_f_t = 0.5*N*(CDF_Xb_f(bin_edges[1:])-CDF_Xb_f(bin_edges[:-1]))
error_N_X_f_t = np.sqrt(expectation_N_X_f_t)
error_N_Xb_f_t = np.sqrt(expectation_N_Xb_f_t)

#Propagate the errors
error_A_CP_t = (2*expectation_N_X_f_t*expectation_N_Xb_f_t)/(expectation_N_X_f_t+expectation_N_Xb_f_t)**2 * np.sqrt((error_N_X_f_t/expectation_N_X_f_t)**2+(error_N_Xb_f_t/expectation_N_Xb_f_t)**2)

#Plot the CP asymmetry of the sample
fig, ax = plt.subplots()
ax.set_title(r"CP Asymmetry of the $X^{0} - X^{0}$ System as a Function of Time")
ax.set_xlabel(r"Decay Time, $\Gamma t$")
ax.set_ylabel("CP Asymmetry")
ax.errorbar(bin_midpoints, A_CP_t, yerr=error_A_CP_t, label="Sample CP Asymmetry", color="#1f77b4", fmt="o", markersize=4, capsize=2)
ax.plot(t_plot_range, A_CP(t_plot_range, C_f, D_f, S_f), label="Expected CP Asymmetry", color="red")
ax.set_xlim(0, 5)
ax.legend(loc="lower left")
plt.show()


# In[11]:


#Repeat the fitting process to check the effect of varying the mis-identification rate omega

#Define CP violation and decay parameters
modulus_q_p = 1.0010
x = 0.7697
y = - 0.0005
C_f = 0.002
S_f = 0.711
D_f = - np.sqrt(1 - C_f**2 - S_f**2)

#Define decay time and invariant mass parameters for signals and background (except for signal decay time parameters CDS, which are defined at the start)
#If p_sig=1, sample has no background events, recommend defining bkg parameters as np.nan
m_sig_mean = 5280
m_sig_width = 30
m_X_f_mean = m_sig_mean
m_X_f_width = m_sig_width
m_Xb_f_mean = m_sig_mean
m_Xb_f_width = m_sig_width
tau_bkg = 1            #Need to adjust t_range in PDF_mixed generation accordingly
m_bkg_mean = 5280
m_bkg_width = 400     #Need to adjust base probability outside uniform distribution range

#Define detector parameters
N_sig_expected = 13615395     #N_sig is a run parameter
p_sig_expected = 0.28     #p_sig is a run parameter
#omega is a run parameter
eps_sig_expected = 0.46
eps_X_f = eps_sig_expected
eps_Xb_f = eps_sig_expected
eps_bkg = 1

#Define run parameters
run_number = 100
N_sig_list = np.array([10**5])
p_sig_list = np.array([p_sig_expected])
omega_list = np.linspace(0, 0.4, 2)

#Initialise arrays for the storage of iminuit results
C_f_value_array = np.zeros((len(N_sig_list), len(p_sig_list), len(omega_list), run_number))
D_f_value_array = np.zeros((len(N_sig_list), len(p_sig_list), len(omega_list), run_number))
S_f_value_array = np.zeros((len(N_sig_list), len(p_sig_list), len(omega_list), run_number))
C_f_error_array = np.zeros((len(N_sig_list), len(p_sig_list), len(omega_list), run_number))
D_f_error_array = np.zeros((len(N_sig_list), len(p_sig_list), len(omega_list), run_number))
S_f_error_array = np.zeros((len(N_sig_list), len(p_sig_list), len(omega_list), run_number))
p_sig_value_array = np.zeros((len(N_sig_list), len(p_sig_list), len(omega_list), run_number))
p_sig_error_array = np.zeros((len(N_sig_list), len(p_sig_list), len(omega_list), run_number))


for N_sig_index in range(len(N_sig_list)):
    for p_sig_index in range(len(p_sig_list)):
        for omega_index in range(len(omega_list)):

            #Note that, while PDF_mixed calls omega, omega is not a variable to be fitted by iminuit, hence it must be defined globally
            N_sig = N_sig_list[N_sig_index]
            p_sig = p_sig_list[p_sig_index]
            omega = omega_list[omega_index]

            print(f"Running: N_sig_index = {N_sig_index+1}/{len(N_sig_list)}, p_sig_index = {p_sig_index+1}/{len(p_sig_list)}, omega_index = {omega_index+1}/{len(omega_list)}")
            time_checkpoint = time.time()

            for run_index in range(run_number):

                #Sample generation and fitting
                #time_checkpoint_generation = time.time()
                sample_t, sample_m, sample_q = sample_generation_simple()
                #print(f"Generation Time = {time.time()-time_checkpoint_generation:.1f} s")
                mi = Minuit(cost_function_simple, C_f=C_f, D_f=D_f, S_f=S_f, p_sig=p_sig)
                #time_checkpoint_fitting = time.time()
                mi.limits["C_f", "S_f"] = (-1, 1)
                mi.fixed["D_f"] = True
                mi.limits["p_sig"] = (0, 1)
                mi.migrad()
                mi.hesse()
                #print(f"Fitting Time = {time.time()-time_checkpoint_fitting:.1f} s")

                #Store the iminuit results
                C_f_value_array[N_sig_index][p_sig_index][omega_index][run_index] = mi.values["C_f"]
                D_f_value_array[N_sig_index][p_sig_index][omega_index][run_index] = mi.values["D_f"]
                S_f_value_array[N_sig_index][p_sig_index][omega_index][run_index] = mi.values["S_f"]
                C_f_error_array[N_sig_index][p_sig_index][omega_index][run_index] = mi.errors["C_f"]
                D_f_error_array[N_sig_index][p_sig_index][omega_index][run_index] = mi.errors["D_f"]
                S_f_error_array[N_sig_index][p_sig_index][omega_index][run_index] = mi.errors["S_f"]
                p_sig_value_array[N_sig_index][p_sig_index][omega_index][run_index] = mi.values["p_sig"]
                p_sig_error_array[N_sig_index][p_sig_index][omega_index][run_index] = mi.errors["p_sig"]

            print(f"Run-Time = {time.time()-time_checkpoint:.1f} s \n")


# In[ ]:


#Fitted paramter value statistics
#Mean
C_f_mean_array = np.mean(C_f_value_array, axis=3)
D_f_mean_array = np.mean(D_f_value_array, axis=3)
S_f_mean_array = np.mean(S_f_value_array, axis=3)
#Width
C_f_width_array = np.std(C_f_value_array, axis=3, ddof=1)
D_f_width_array = np.std(D_f_value_array, axis=3, ddof=1)
S_f_width_array = np.std(S_f_value_array, axis=3, ddof=1)
#Error in mean
C_f_mean_error_array = C_f_width_array/np.sqrt(run_number)
D_f_mean_error_array = D_f_width_array/np.sqrt(run_number)
S_f_mean_error_array = S_f_width_array/np.sqrt(run_number)
#Error in width
C_f_width_error_array = C_f_width_array/np.sqrt(2*(run_number-1))
D_f_width_error_array = D_f_width_array/np.sqrt(2*(run_number-1))
S_f_width_error_array = S_f_width_array/np.sqrt(2*(run_number-1))

#Fitted parameter pull statistics
#Compute the pull
C_f_pull_array = (C_f_value_array-C_f)/C_f_error_array
D_f_pull_array = (D_f_value_array-D_f)/D_f_error_array
S_f_pull_array = (S_f_value_array-S_f)/S_f_error_array
#Mean
C_f_pull_mean_array = np.mean(C_f_pull_array, axis=3)
D_f_pull_mean_array = np.mean(D_f_pull_array, axis=3)
S_f_pull_mean_array = np.mean(S_f_pull_array, axis=3)
#Width
C_f_pull_width_array = np.std(C_f_pull_array, axis=3, ddof=1)
D_f_pull_width_array = np.std(D_f_pull_array, axis=3, ddof=1)
S_f_pull_width_array = np.std(S_f_pull_array, axis=3, ddof=1)
#Error in mean
C_f_pull_mean_error_array = C_f_pull_width_array/np.sqrt(run_number)
D_f_pull_mean_error_array = D_f_pull_width_array/np.sqrt(run_number)
S_f_pull_mean_error_array = S_f_pull_width_array/np.sqrt(run_number)

#Error in width
C_f_pull_width_error_array = C_f_pull_width_array/np.sqrt(2*(run_number-1))
D_f_pull_width_error_array = D_f_pull_width_array/np.sqrt(2*(run_number-1))
S_f_pull_width_error_array = S_f_pull_width_array/np.sqrt(2*(run_number-1))





#Angle beta statistics
#Compute angle beta
beta = 180/np.pi * 0.5 * np.arcsin(S_f)
beta_value_array = 180/np.pi * 0.5 * np.arcsin(S_f_value_array)
beta_error_array = 180/np.pi * 0.5 * 1/np.sqrt(1-S_f_value_array**2) * S_f_error_array
#Mean
beta_mean_array = np.mean(beta_value_array, axis=3)
#Width
beta_width_array = np.std(beta_value_array, axis=3, ddof=1)
#Error in mean
beta_mean_error_array = beta_width_array/np.sqrt(run_number)
#Error in width
beta_width_error_array = beta_width_array/np.sqrt(2*(run_number-1))

#Angle beta pull statistics
#Compute the pull
beta_pull_array = (beta_value_array-beta)/beta_error_array
#Mean
beta_pull_mean_array = np.mean(beta_pull_array, axis=3)
#Width
beta_pull_width_array = np.std(beta_pull_array, axis=3, ddof=1)
#Error in mean
beta_pull_mean_error_array = beta_pull_width_array/np.sqrt(run_number)
#Error in width
beta_pull_width_error_array = beta_pull_width_array/np.sqrt(2*(run_number-1))

#Scale results to the expected number of initial signals
for N_sig_index in range(len(N_sig_list)):
    beta_mean_error_array[N_sig_index] = 1/(N_sig_expected/N_sig_list[N_sig_index]) * beta_mean_error_array[N_sig_index]
    beta_width_array[N_sig_index] = 1/np.sqrt(N_sig_expected/N_sig_list[N_sig_index]) * beta_width_array[N_sig_index]
    beta_width_error_array[N_sig_index] = 1/np.sqrt(N_sig_expected/N_sig_list[N_sig_index]) * beta_width_error_array[N_sig_index]





#Fitted signal fraction statistics
#Mean
p_sig_mean_array = np.mean(p_sig_value_array, axis=3)
#Width
p_sig_width_array = np.std(p_sig_value_array, axis=3, ddof=1)
#Error in mean
p_sig_mean_error_array = p_sig_width_array/np.sqrt(run_number)
#Error in width
p_sig_width_error_array = p_sig_width_array/np.sqrt(2*(run_number-1))

#Fitted signal fraction pull statistics
#Compute the pull
p_sig_pull_array = np.zeros((len(N_sig_list), len(p_sig_list), len(omega_list), run_number))
for N_sig_index in range(len(N_sig_list)):
    for p_sig_index in range(len(p_sig_list)):
        for omega_index in range(len(omega_list)):
            for run_index in range(run_number):
                p_sig_pull_array[N_sig_index][p_sig_index][omega_index][run_index] = (p_sig_value_array[N_sig_index][p_sig_index][omega_index][run_index] - p_sig_list[p_sig_index])/p_sig_error_array[N_sig_index][p_sig_index][omega_index][run_index]
#Mean
p_sig_pull_mean_array = np.mean(p_sig_pull_array, axis=3)
#Width
p_sig_pull_width_array = np.std(p_sig_pull_array, axis=3, ddof=1)
#Error in mean
p_sig_pull_mean_error_array = p_sig_pull_width_array/np.sqrt(run_number)
#Error in width
p_sig_pull_width_error_array = p_sig_pull_width_array/np.sqrt(2*(run_number-1))


# In[ ]:


#Plot fitted parameter values (mean +/- width) as a function of the different mis-identification rate
fig, ax = plt.subplots()
ax.set_title("Fitted Parameter Values as Functions of Mis-Identification Rate")
ax.set_xlabel(r"Mis-Identification Rate, $\omega$")
ax.set_ylabel("Fitted Parameter Value")
for N_sig_index in range(len(N_sig_list)):
    for p_sig_index in range(len(p_sig_list)):
        #ax.errorbar(omega_list, C_f_mean_array[N_sig_index][p_sig_index], yerr=C_f_width_array[N_sig_index][p_sig_index], fmt="o", markersize=4, capsize=2, label=rf"Fitted $C_{{f}}$ value; N_sig = {N_sig_list[N_sig_index]:.1e}, p_sig = {p_sig_list[p_sig_index]}")
        ax.errorbar(omega_list, S_f_mean_array[N_sig_index][p_sig_index], yerr=S_f_width_array[N_sig_index][p_sig_index], fmt="o", markersize=4, capsize=2, label=rf"Fitted $S_{{f}}$ value; N_sig = {N_sig_list[N_sig_index]:.1e}, p_sig = {p_sig_list[p_sig_index]}")
ax.plot(omega_list, S_f*np.ones(len(omega_list)), color="black", label=rf"True $S_{{f}}$ = {S_f}")
ax.legend(loc="upper left")
plt.show()

#Plot fitted parameter widths (width +/- width_error) as a function of the different mis-identification rate
fig, ax = plt.subplots()
ax_twin = ax.twinx()
ax.set_title("Fitted Parameter Widths as Functions of Mis-Identification Rate")
ax.set_xlabel(r"Mis-Identification Rate, $\omega$")
ax.set_ylabel("Fitted Parameter Width")
for N_sig_index in range(len(N_sig_list)):
    for p_sig_index in range(len(p_sig_list)):
        #ax.errorbar(omega_list, C_f_width_array[N_sig_index][p_sig_index], yerr=C_f_width_error_array[N_sig_index][p_sig_index], fmt="o", markersize=4, capsize=2, label=rf"Fitted $C_{{f}}$ width; N_sig = {N_sig_list[N_sig_index]:.1e}, p_sig = {p_sig_list[p_sig_index]}")
        ax.errorbar(omega_list, S_f_width_array[N_sig_index][p_sig_index], yerr=S_f_width_error_array[N_sig_index][p_sig_index], fmt="o", markersize=4, capsize=2, label=rf"Fitted $S_{{f}}$ width; N_sig = {N_sig_list[N_sig_index]:.1e}, p_sig = {p_sig_list[p_sig_index]}")
ax_twin.plot(omega_list, 1/(1-2*omega_list), color="black", label=r"Expected behaviour 1/(1-2$\omega$)")
ax_twin.get_yaxis().set_visible(False)
ax.legend(loc="upper left")
ax_twin.legend(loc="center left")
plt.show()


#Plot parameter pull mean (mean +/- mean error) as a function of the different mis-identification rate
fig, ax = plt.subplots()
ax.set_title("Fitted Parameter Pull Means as Functions of Mis-Identification Rate")
ax.set_xlabel(r"Mis-Identification Rate, $\omega$")
ax.set_ylabel("Fitted Parameter Pull Mean")
for N_sig_index in range(len(N_sig_list)):
    for p_sig_index in range(len(p_sig_list)):
        #ax.errorbar(omega_list, C_f_pull_mean_array[N_sig_index][p_sig_index], yerr=C_f_pull_mean_error_array[N_sig_index][p_sig_index], fmt="o", markersize=4, capsize=2, label=rf"Fitted $C_{{f}}$ pull mean; N_sig = {N_sig_list[N_sig_index]:.1e}, p_sig = {p_sig_list[p_sig_index]}")
        ax.errorbar(omega_list, S_f_pull_mean_array[N_sig_index][p_sig_index], yerr=S_f_pull_mean_error_array[N_sig_index][p_sig_index], fmt="o", markersize=4, capsize=2, label=rf"Fitted $S_{{f}}$ pull mean; N_sig = {N_sig_list[N_sig_index]:.1e}, p_sig = {p_sig_list[p_sig_index]}")
ax.plot(omega_list, np.zeros(len(omega_list)), color="black")
ax.legend(loc="upper left")
plt.show()

#Plot parameter pull width (width +/- width error) as a function of the different mis-identification rate
fig, ax = plt.subplots()
ax.set_title("Fitted Parameter Pull Widths as Functions of Mis-Identification Rate")
ax.set_xlabel(r"Mis-Identification Rate, $\omega$")
ax.set_ylabel("Fitted Parameter Pull Width")
for N_index in range(len(N_sig_list)):
    for p_sig_index in range(len(p_sig_list)):
        #ax.errorbar(omega_list, C_f_pull_width_array[N_sig_index][p_sig_index], yerr=C_f_pull_width_error_array[N_sig_index][p_sig_index], fmt="o", markersize=4, capsize=2, label=rf"Fitted $C_{{f}}$ pull width; N_sig = {N_sig_list[N_sig_index]:.1e}, p_sig = {p_sig_list[p_sig_index]}")
        ax.errorbar(omega_list, S_f_pull_width_array[N_sig_index][p_sig_index], yerr=S_f_pull_width_error_array[N_sig_index][p_sig_index], fmt="o", markersize=4, capsize=2, label=rf"Fitted $S_{{f}}$ pull width; N_sig = {N_sig_list[N_sig_index]:.1e}, p_sig = {p_sig_list[p_sig_index]}")
ax.plot(omega_list, np.ones(len(omega_list)), color="black")
ax.legend(loc="upper left")
plt.show()





#Plot angle beta values (mean +/- width) as a function of the different mis-identification rate
fig, ax = plt.subplots()
ax.set_title(r"Fitted Angle $\beta$ Values as Functions of Mis-Identification Rate")
ax.set_xlabel(r"Mis-Identification Rate, $\omega$")
ax.set_ylabel(r"Angle $\beta$ Value (Degree)")
for N_sig_index in range(len(N_sig_list)):
    for p_sig_index in range(len(p_sig_list)):
        ax.errorbar(omega_list, beta_mean_array[N_sig_index][p_sig_index], yerr=beta_width_array[N_sig_index][p_sig_index], fmt="o", markersize=4, capsize=2, label=rf"Fitted $\beta$ value; scaled from N_sig = {N_sig_list[N_sig_index]:.1e}, p_sig = {p_sig_list[p_sig_index]}")
ax.plot(omega_list, 22.6174577*np.ones(len(omega_list)), linestyle="dashed", color="black")
ax.plot(omega_list, beta*np.ones(len(omega_list)), linestyle="dashed", color="blue")
ax.axhspan(22.6174577+0.447495083, 22.6174577-0.447495083, color='red', alpha=0.5, label=r"World average $\beta = 22.6 \pm 0.4 \degree$")
ax.legend(loc="lower left")
plt.show()

#Plot angle beta width (width +/- width_error) as a function of the different mis-identification rate
fig, ax = plt.subplots()
ax_twin = ax.twinx()
ax.set_title(r"Fitted Angle $\beta$ Widths as Functions of Mis-Identification Rate")
ax.set_xlabel(r"Mis-Identification Rate, $\omega$")
ax.set_ylabel(r"Angle $\beta$ Width (Degree)")
for N_sig_index in range(len(N_sig_list)):
    for p_sig_index in range(len(p_sig_list)):
        ax.errorbar(omega_list, beta_width_array[N_sig_index][p_sig_index], yerr=beta_width_error_array[N_sig_index][p_sig_index], fmt="o", markersize=4, capsize=2, label=rf"Fitted $\beta$ width; scaled from N_sig = {N_sig_list[N_sig_index]:.1e}, p_sig = {p_sig_list[p_sig_index]}")
#ax.plot(omega_list, 0.447495083*np.ones(len(omega_list)), label=r"World Average $\beta$ Width = $0.45 \degree$")
ax_twin.plot(omega_list, 1/(1-2*omega_list), color="black", label=r"Expected behaviour 1/(1-2$\omega$)")
ax_twin.get_yaxis().set_visible(False)
ax.legend(loc="upper left")
ax_twin.legend(loc="center left")
plt.show()


#Plot angle beta pull mean (mean +/- mean error) as a function of the different mis-identification rate
fig, ax = plt.subplots()
ax.set_title(r"Fitted Angle $\beta$ Pull Means as Functions of Mis-Identification Rate")
ax.set_xlabel(r"Mis-Identification Rate, $\omega$")
ax.set_ylabel(r"Fitted Angle $\beta$ Pull Mean")
for N_sig_index in range(len(N_sig_list)):
    for p_sig_index in range(len(p_sig_list)):
        ax.errorbar(omega_list, beta_pull_mean_array[N_sig_index][p_sig_index], yerr=beta_pull_mean_error_array[N_sig_index][p_sig_index], fmt="o", markersize=4, capsize=2, label=rf"Fitted angle $\beta$ pull mean; N_sig = {N_sig_list[N_sig_index]:.1e}, p_sig = {p_sig_list[p_sig_index]}")
ax.plot(omega_list, np.zeros(len(omega_list)), color="black")
ax.legend(loc="upper left")
plt.show()

#Plot angle beta pull width (width +/- width error) as a function of the different mis-identification rate
fig, ax = plt.subplots()
ax.set_title(r"Fitted Angle $\beta$ Pull Widths as Functions of Mis-Identification Rate")
ax.set_xlabel(r"Mis-Identification Rate, $\omega$")
ax.set_ylabel(r"Fitted Angle $\beta$ Pull Width")
for N_sig_index in range(len(N_sig_list)):
    for p_sig_index in range(len(p_sig_list)):
        ax.errorbar(omega_list, beta_pull_width_array[N_sig_index][p_sig_index], yerr=beta_pull_width_error_array[N_sig_index][p_sig_index], fmt="o", markersize=4, capsize=2, label=rf"Fitted angle $\beta$ pull width; N_sig = {N_sig_list[N_sig_index]:.1e}, p_sig = {p_sig_list[p_sig_index]}")
ax.plot(omega_list, np.ones(len(omega_list)), color="black")
ax.legend(loc="upper left")
plt.show()





#Plot fitted signal fraction values (mean +/- width) as a function of the different mis-identification rate
fig, ax = plt.subplots()
ax.set_title("Fitted Signal Fraction Values as Functions of Mis-Identification Rate")
ax.set_xlabel(r"Mis-Identification Rate, $\omega$")
ax.set_ylabel("Fitted Signal Fraction Value")
for N_sig_index in range(len(N_sig_list)):
    for p_sig_index in range(len(p_sig_list)):
        ax.errorbar(omega_list, p_sig_mean_array[N_sig_index][p_sig_index], yerr=p_sig_width_array[N_sig_index][p_sig_index], fmt="o", markersize=4, capsize=2, label=rf"Fitted $p_{{sig}}$ value; N_sig = {N_sig_list[N_sig_index]:.1e}, $p_{{sig}}$ = {p_sig_list[p_sig_index]}")
        ax.plot(omega_list, p_sig_list[p_sig_index]*np.ones(len(omega_list)), color="black", label=rf"True $p_{{sig}}$ = {p_sig_list[p_sig_index]}")
ax.legend(loc="upper left")
plt.show()

#Plot fitted signal fraction widths (width +/- width_error) as a function of the different mis-identification rate
fig, ax = plt.subplots()
#ax_twin = ax.twinx()
ax.set_title("Fitted Signal Fraction Widths as Functions of Mis-Identification Rate")
ax.set_xlabel(r"Mis-Identification Rate, $\omega$")
ax.set_ylabel("Fitted Signal Fraction Width")
for N_sig_index in range(len(N_sig_list)):
    for p_sig_index in range(len(p_sig_list)):
        ax.errorbar(omega_list, p_sig_width_array[N_sig_index][p_sig_index], yerr=p_sig_width_error_array[N_sig_index][p_sig_index], fmt="o", markersize=4, capsize=2, label=rf"Fitted $p_{{sig}}$ width; N_sig = {N_sig_list[N_sig_index]:.1e}, $p_{{sig}}$ = {p_sig_list[p_sig_index]}")
#ax_twin.plot(omega_list, 1/(1-2*omega_list), color="black", label=r"Expected behaviour 1/(1-2$\omega$)")
#ax_twin.get_yaxis().set_visible(False)
ax.legend(loc="upper left")
#ax_twin.legend(loc="center left")
plt.show()


#Plot signal fraction pull mean (mean +/- mean error) as a function of the different mis-identification rate
fig, ax = plt.subplots()
ax.set_title("Fitted Signal Fraction Pull Means as Functions of Mis-Identification Rate")
ax.set_xlabel(r"Mis-Identification Rate, $\omega$")
ax.set_ylabel("Fitted Signal Fraction Pull Mean")
for N_sig_index in range(len(N_sig_list)):
    for p_sig_index in range(len(p_sig_list)):
        ax.errorbar(omega_list, p_sig_pull_mean_array[N_sig_index][p_sig_index], yerr=p_sig_pull_mean_error_array[N_sig_index][p_sig_index], fmt="o", markersize=4, capsize=2, label=rf"Fitted $p_{{sig}}$ pull mean; N_sig = {N_sig_list[N_sig_index]:.1e}, $p_{{sig}}$ = {p_sig_list[p_sig_index]}")
ax.plot(omega_list, np.zeros(len(omega_list)), color="black")
ax.legend(loc="upper left")
plt.show()

#Plot signal fraction pull width (width +/- width error) as a function of the different mis-identification rate
fig, ax = plt.subplots()
ax.set_title("Fitted Signal Fraction Pull Widths as Functions of Mis-Identification Rate")
ax.set_xlabel(r"Mis-Identification Rate, $\omega$")
ax.set_ylabel("Fitted Signal Fraction Pull Width")
for N_sig_index in range(len(N_sig_list)):
    for p_sig_index in range(len(p_sig_list)):
        ax.errorbar(omega_list, p_sig_pull_width_array[N_sig_index][p_sig_index], yerr=p_sig_pull_width_error_array[N_sig_index][p_sig_index], fmt="o", markersize=4, capsize=2, label=rf"Fitted $p_{{sig}}$ pull width; N_sig = {N_sig_list[N_sig_index]:.1e}, $p_{{sig}}$ = {p_sig_list[p_sig_index]}")
ax.plot(omega_list, np.ones(len(omega_list)), color="black")
ax.legend(loc="upper left")
plt.show()


# In[ ]:


#Save raw data
#np.save("run_0_C_f_value_array.npy", C_f_value_array)
#np.save("run_0_D_f_value_array.npy", D_f_value_array)
#np.save("run_0_S_f_value_array.npy", S_f_value_array)
#np.save("run_0_C_f_error_array.npy", C_f_error_array)
#np.save("run_0_D_f_error_array.npy", D_f_error_array)
#np.save("run_0_S_f_error_array.npy", S_f_error_array)
#np.save("run_0_p_sig_value_array.npy", p_sig_value_array)
#np.save("run_0_p_sig_error_array.npy", p_sig_error_array)

##np.load("data.npy")

