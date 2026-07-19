import os
import matplotlib.pyplot as plt
import numpy as np

def generate_csv(chk,i):
    
    fieldconvert = f"FieldConvert -f -m interppoints:fromxml=SodTubeSession.xml:fromfld={chk}:line=201,-1,0,1,0 points_{i}.csv"
    os.system(fieldconvert)


for i in range(5):

    generate_csv(f"SodTubeSession_{i}.chk", i)
    data = np.genfromtxt(f'points_{i}.csv', delimiter=',', skip_header=1)

    fig, ax = plt.subplots(2,3)

    ax[0,0].plot(data[:,0], data[:,2])
    ax[0,0].set_xlabel(r'$x$')
    ax[0,0].set_ylabel(r'$\rho$')
    ax[0,0].grid()

    ax[0,1].plot(data[:,0], data[:,5])
    ax[0,1].set_xlabel(r'$x$')
    ax[0,1].set_ylabel(r'$u$')
    ax[0,1].grid()

    ax[1,0].plot(data[:,0], data[:,6])
    ax[1,0].set_xlabel(r'$x$')
    ax[1,0].set_ylabel(r'$P$')
    ax[1,0].grid()

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

    figname = f'plot_{i}.png'
    print(f'Saving {figname}')
    fig.set_size_inches(12,6)
    fig.subplots_adjust(wspace=0.3, hspace=0.5)
    fig.savefig(f'plot_{i}.png', dpi=200, bbox_inches='tight')

    plt.close('all')
    print(f'Done')
