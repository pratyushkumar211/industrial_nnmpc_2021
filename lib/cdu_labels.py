"""
Script to store the 32 input labels, the 90 output labels,
the 4 controlled output labels, and the 3 disturbance labels
for the crude distillation unit example. 
"""

ulabels = [f'$u_{{{i+1}}}$' for i in range(32)]

ylabels = [f'$y_{{{i+1}}}$' for i in range(90)]

zlabels = ['NAPHTHA', 
           'KEROSENE-FLASH',
           'KEROSENE', 
           'DIESEL']

pdlabels = ['$p_1 (\Delta u_{1})$',
            '$p_2 (\Delta u_{4})$', 
            '$p_3 (\Delta u_{5})$',
            '$p_4 (\Delta u_{6})$',
            '$p_5 (\Delta u_{7})$', 
            '$p_6 (\Delta u_{12})$', 
            '$p_7 (\Delta u_{19})$', 
            '$p_8 (\Delta u_{20})$',
            '$p_9 (\Delta u_{24})$', 
            '$p_{10} (\Delta u_{26})$', 
            '$p_{11} (\Delta u_{28})$', 
            '$p_{12} (\Delta u_{31})$',
            '$p_{13} (\Delta u_{32})$',
            '$p_{11} (\Delta u_{28})$', 
            '$p_{12} (\Delta u_{31})$',
            '$p_{13} (\Delta u_{32})$']

