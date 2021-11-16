import streamlit as st

import numpy as np
import pandas as pd

from scipy.stats import beta
import matplotlib.pyplot as plt


from scipy.optimize import fmin
#from scipy.stats import *

def HDIofICDF(dist_name, credMass=0.95, **args):
    # freeze distribution with given arguments
    distri = dist_name(**args)
    # initial guess for HDIlowTailPr
    incredMass =  1.0 - credMass

    def intervalWidth(lowTailPr):
        return distri.ppf(credMass + lowTailPr) - distri.ppf(lowTailPr)

    # find lowTailPr that minimizes intervalWidth
    HDIlowTailPr = fmin(intervalWidth, incredMass, ftol=1e-8, disp=False)[0]
    # return interval as array([low, high])
    return distri.ppf([HDIlowTailPr, credMass + HDIlowTailPr])


def counts_to_df_success(success, failure, a_prior=0, b_prior=0, hdi_range=0.95, d_psuccess = 0.0001, 
                           min_psuccess=0.85, max_psucess=1.):
    a = success + a_prior
    b = failure + b_prior
    
    hdi_min, hdi_max = HDIofICDF(beta, a=a, b=b, credMass=hdi_range)
    
    
    p_success = np.arange(min_psuccess, max_psucess + d_psuccess, step=d_psuccess)
    
    df_success = pd.DataFrame({"p_success": p_success, "beta_pdf": beta.pdf(p_success, a, b)}).set_index("p_success")
    df_success["HDI_range"] = (df_success.index >= hdi_min) & (df_success.index <= hdi_max)
    
    return df_success


def plot_success_rates(df, color="purple", label=None, fill=False, plot_hdi=False, alpha=1.):
    query_hdi = "HDI_range"
    
    # label = f"HDI {hdi_range * 100:0.0f}%"
    
    plt.plot(df.index * 100., df["beta_pdf"], '-',linewidth=3, color=color, label=label, alpha=alpha)
    
    
    if plot_hdi:
        if fill:
            plt.fill_between(df.query(query_hdi).index * 100., df.query(query_hdi)["beta_pdf"], '-',linewidth=3, color=color, alpha=0.4)
        else:
            min_hdi = df.query(query_hdi).index.min()
            max_hdi = df.query(query_hdi).index.max()
            height = np.min([df.loc[min_hdi, "beta_pdf"], 
                      df.loc[max_hdi, "beta_pdf"] ])
            
            
            plt.plot([min_hdi * 100, max_hdi * 100],  [height, height], "--",color=color )

"""
Hi!
"""






prior_factor_default = 1/2.
min_value = 0.85
max_value = 1.

p_success = st.sidebar.slider('True Safety', min_value=min_value, max_value=max_value, value=0.94)
n_cards = st.sidebar.number_input('Number of cases', min_value=20, max_value=1000, step=10, value=100)

p_prior = st.sidebar.slider('Prior Safety', min_value=min_value, max_value=max_value, value=0.96)
n_prior = st.sidebar.number_input('Number of prior cases', min_value=20, max_value=10000, step=10, value=100) 
prior_factor = st.sidebar.slider('Prior weight', min_value=0., max_value=1., value=prior_factor_default, step=0.1)


sr_counts = {True: n_cards * p_success, False: n_cards * (1 - p_success)}
sr_counts_prior = {True: n_prior * p_prior, False: n_prior * (1 - p_prior)}

sr_counts_prior

# calculating beta pdf
df_wout_prior = counts_to_df_success(sr_counts[True], sr_counts[False]) 
df_prior = counts_to_df_success(sr_counts_prior[True] * prior_factor, sr_counts_prior[False] * prior_factor) 
df_with_prior = counts_to_df_success(sr_counts[True], sr_counts[False], 
                                     a_prior=sr_counts_prior[True] * prior_factor, 
                                     b_prior=sr_counts_prior[False] * prior_factor)  

sr_counts_prior[True] * prior_factor

sr_counts_prior[False] * prior_factor



st.write(df_prior.sort_values("beta_pdf", ascending=False))

# plotting 
fig = plt.figure()
plot_success_rates(df_wout_prior, color="red", fill=False, plot_hdi=True, label="no prior")
plot_success_rates(df_prior, color="gray", plot_hdi=False, label="prior", alpha=0.4)
plot_success_rates(df_with_prior, color="green", fill=True, plot_hdi=True, label="with prior")
plt.legend()

st.pyplot(fig)
