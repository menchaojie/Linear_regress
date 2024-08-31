import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 4 * np.pi, 100)
y = np.sin(x)

plt.plot(x, y)
plt.fill_between(x, y+0.5, y-0.5, alpha=0.1, color='r')

plt.show()
