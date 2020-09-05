"""
Script to store the 6 input labels, the 6 output labels,
the 12 state output labels, and the 5 disturbance labels
for the cstrs with flash example. 
"""

ylabels = ['$H_{r}$', '$x_{Ar}$', '$x_{Br}$', '$T_{r}$',
           '$H_{m}$', '$x_{Am}$', '$x_{Bm}$', '$T_{m}$',
           '$H_{b}$', '$x_{Ab}$', '$x_{Bb}$', '$T_{b}$']

zlabels = ['$H_{r}$ (m)', '$T_{r}$ (K)',
           '$H_{m}$ (m)', '$T_{m}$ (K)',
           '$H_{b}$ (m)', '$T_{b}$ (K)']

ulabels = ['$F_{0}$ (Kg/s)', '$Q_{r} \ (10^3$ kW)',
           '$F_{1}$ (Kg/s)', '$Q_{m} \ (10^3$ kW)',
           '$D$ (Kg/s)', '$Q_{b} \ (10^3$ kW)']

pdlabels = ['$x_{A0}$', '$x_{B0}$',
           '$x_{A1}$', '$x_{B1}$',
           '$T_{0}$']