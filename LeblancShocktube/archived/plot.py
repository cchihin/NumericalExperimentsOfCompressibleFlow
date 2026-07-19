import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt('solution1D.txt')

fig, ax = plt.subplots()

ax.plot(data[:,0], data[:,1])
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\rho$')
ax.grid()

fig.savefig('plot.pdf', bbox_inches='tight')
