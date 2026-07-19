import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
from LeblancSolution import leblanc_exact

def generate_csv(path, session, chk, i):
    
    fieldconvert = f"FieldConvert -f -m interppoints:fromxml={session}:fromfld={chk}:line=201,0,0,10,0 points_{i}.csv"
    subprocess.run(fieldconvert.split(), cwd=path)

TFinal = 35
P = 4
Nel = 80
mu0 = 1

# Processing 
# for t in range(TFinal):
#     generate_csv(f".", f"Leblanc_{P}.xml", f"Leblanc_{P}_{t}.chk", t)

# Plotting
for t in range(TFinal):

    fig, ax = plt.subplots(2,3)

    x = np.linspace(0, 10, 1000)

    rho, u_vel, p, E = leblanc_exact(x, 3, t*0.1)
    
    data = np.genfromtxt(f'points_{t}.csv', delimiter=',', skip_header=1)

    ax[0,0].plot(x, rho, 'k--')
    ax[0,0].plot(data[:,0], data[:,2], label=f"P={P}")
    ax[0,0].set_xlabel(r'$x$')
    ax[0,0].set_ylabel(r'$\rho$')
    ax[0,0].grid()

    ax[0,1].plot(x, u_vel, 'k--')
    ax[0,1].plot(data[:,0], data[:,5])
    ax[0,1].set_xlabel(r'$x$')
    ax[0,1].set_ylabel(r'$u$')
    ax[0,1].grid()

    ax[1,0].plot(x, p, 'k--')
    ax[1,0].plot(data[:,0], data[:,6])
    ax[1,0].set_xlabel(r'$x$')
    ax[1,0].set_ylabel(r'$P$')
    ax[1,0].grid()

    ax[1,1].plot(x, E, 'k--')
    ax[1,1].plot(data[:,0], data[:,4])
    ax[1,1].set_xlabel(r'$x$')
    ax[1,1].set_ylabel(r'$E$')
    ax[1,1].grid()

    ax[0,2].plot(data[:,0], data[:,-1])
    ax[0,2].set_xlabel(r'$x$')
    ax[0,2].set_ylabel(r'$Art. Vis.$')
    ax[0,2].grid()

    ax[1,2].plot(data[:,0], data[:,-2])
    ax[1,2].set_xlabel(r'$x$')
    ax[1,2].set_ylabel(r'$Sensor$')
    ax[1,2].grid()

    ax[0,0].legend(ncol=3, loc="lower left", bbox_to_anchor=(1.2,1))
    figname = f'plot_{t}.png'
    fig.suptitle(fr'$N_{{el}} = {Nel}, t = {t:.1f}, \mu_0 = {mu0}$')
    print(f'Saving {figname}')
    fig.set_size_inches(12,6)
    fig.subplots_adjust(wspace=0.3, hspace=0.5)
    fig.savefig(f'plot_{t}.png', dpi=200, bbox_inches='tight')
    plt.close('all')
    print(f'Done')

        # Plotting
