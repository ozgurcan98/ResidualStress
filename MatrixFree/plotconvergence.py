import matplotlib.pyplot as plt
import numpy as np

# Spatial convergence: (64 bit)
Deltax   = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125])
L2_error = np.array([0.0932505, 0.023294, 0.005823, 0.0014558, 0.0003639, 9.0986e-05, 2.274564e-05])

plt.plot(Deltax, L2_error, marker='o', label='L2 error')
plt.plot([0.2,0.02],[0.01,0.0001],'--',label='Reference 1')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Mesh size (Deltax)')
plt.ylabel('L2 error')
plt.title('Spatial Convergence')
plt.grid()
plt.legend()
plt.show()

