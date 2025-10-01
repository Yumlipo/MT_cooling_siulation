import numpy as np
import matplotlib.pyplot as plt

dir = "Y:\\Documents\\LAB\\MT_cooling\\simulation\\2025.06.04_MT_simulation_1_15uM_10s_cooling\\_32C\\1temp_len_cap_cat.txt"
data = np.loadtxt(dir, skiprows=1).T
print(data)
cap = data[3][np.where(data[3] < 3000)[0]]
k_for_cap = data[0][np.where(data[3] < 3000)[0]]

# Plotting
fig = plt.figure(figsize=(16, 12))
ax = []
ax += [fig.add_subplot(311)]
# ax[-1].scatter(data[0], data[2], label='MT')
ax[-1].scatter(k_for_cap, cap, label='cap')
ax[-1].set_xlim(0, np.max(data[0])+0.1)
ax[-1].set_ylabel('cap size')
ax[-1].legend()
ax[-1].grid(True)

ax += [fig.add_subplot(312, sharex=ax[0])]
ax[-1].scatter(data[0], data[4])
ax[-1].set_ylabel('catastrophes amount')
ax[-1].set_xlabel('k_hydr')
ax[-1].set_xticks(np.arange(0, np.max(data[0])+0.1, 0.1))
ax[-1].grid(True)

ax += [fig.add_subplot(313)]
ax[-1].scatter(cap[np.where(cap < 200)[0]], data[4][np.where(data[3] < 200)[0]])
ax[-1].set_ylabel('catastrophes amount')
ax[-1].set_xlabel('cap size [0:200]')

ax[-1].grid(True)
plt.show()
