import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os, glob, time, re
import matplotlib.ticker as ticker

with open("Y:\\Documents\\LAB\\MT_cooling\\simulation\\2025.06.05_MT_simulation_1_15uM_10s_cooling\\results.txt", 'r') as res_f:
    data = np.loadtxt(res_f, usecols=range(0, 7))
    # print(data)
    #len slope temp cur_temp


temp_simulation = np.array([])
cap_simulation = np.array([])

for filename in glob.glob("Y:\\Documents\\LAB\\MT_cooling\\simulation\\2025.06.05_MT_simulation_1_15uM_10s_cooling\\*\\temp_len_cap_cat.txt", recursive=True):
    print("Open now", filename)
    with open(filename, 'r') as f:
        data_from_simulation = np.loadtxt(f)
        temp_simulation = np.append(temp_simulation, data_from_simulation[:, 1])
        cap_simulation = np.append(cap_simulation, data_from_simulation[:, 3]*8/13)

# print("simulation", temp_simulation, cap_simulation)

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

nans, x = nan_helper(data)
data[nans] = np.interp(x(nans), x(~nans), data[~nans])

len1 = data[:, 1]
len2 = data[:, 2]
len_h1 = data[:, 3]
len_h2 = data[:, 4]
temp = np.round(data[:, 6])
# print(temp)

cap_measurement = len1*((len_h1+len_h2)/len2)*10*8/13

# x = 32 - data[:, 5]
x = 32 - temp
x_simulation = 32 - temp_simulation

std_error_simulation = []
std_error_measurment = []

for i in range(len(temp)//10):
    std_error_simulation += [np.std(cap_simulation[i*10:i*10+9], ddof=1) / np.sqrt(10)]
    std_error_measurment += [np.std(cap_measurement[i*10:i*10+9], ddof=1) / np.sqrt(10)]


res_len = stats.linregress(x, cap_measurement)
res_cap = stats.linregress(x_simulation, cap_simulation)
res_corr = stats.linregress(cap_simulation, cap_measurement)
# res_slope = stats.linregress(x, len2*750)
print("measured cap: y=", round(res_len.intercept, 2), "+", round(res_len.slope, 2), "x")
print("real cap : y=", round(res_cap.intercept, 2), "+", round(res_cap.slope, 2), "x")
print("cap size correlation: y=", round(res_corr.intercept, 2), "+", round(res_corr.slope, 2), "x")
# print("slope", res_slope.intercept + res_slope.slope*x2)
y_measurements = res_len.intercept + res_len.slope*x[::10]
y_simulation = res_cap.intercept + res_cap.slope*x_simulation[::10]
y_corr = res_corr.intercept + res_corr.slope*cap_simulation



fig = plt.figure(figsize=(12, 8))
ax = []

ax += [fig.add_subplot(311)]
ax[0].scatter(x, cap_measurement, label="cap from measurements")
# xax = ax[0].axes.get_xaxis()
# xax = xax.set_visible(False)
ax[0].plot(x, res_len.intercept + res_len.slope*x, color='pink', label='fitted len for measured cap')
# ax[0].errorbar(x[::10], y_measurements - std_error_measurment, y_measurements + std_error_measurment, alpha=0.2)
ax[0].errorbar(x[::10], y_measurements, std_error_measurment, fmt='o', capsize=5, alpha=1)
ax[0].xaxis.set_major_locator(ticker.MultipleLocator(2))
ax[0].set_xlim(20, 44)
ax[0].set_xlabel('delta temperature (C)')
ax[0].set_ylabel('measured cap size')


ax += [fig.add_subplot(312, sharex=ax[0])]
ax[1].scatter(x_simulation, cap_simulation, label="real cap size", color="green")
ax[1].plot(x_simulation, res_cap.intercept + res_cap.slope*x_simulation, color='orange', label='fitted real cap size')
# plt.errorbar(x, y, yerr=y_err, fmt='o-', capsize=5, label='Data ± Error')
ax[1].errorbar(x_simulation[::10], y_simulation, std_error_simulation, fmt='o', capsize=5, alpha=1)
ax[1].set_xlabel('delta temperature (C)')
ax[1].set_ylabel('real cap size')

ax += [fig.add_subplot(313)]
ax[2].scatter(cap_simulation, cap_measurement, label="simulation vs. measurements", color="black")
ax[2].plot(cap_simulation, res_corr.intercept + res_corr.slope*cap_simulation, color='red', label='cap size correlation')
# ax[2].fill_between(x, y_corr - std_error_measurment, y_measurements + std_error_measurment, alpha=0.2)
ax[2].set_xlabel('real cap size')
ax[2].set_ylabel('measured cap size')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.legend()
plt.show()

# fig, ax1 = plt.subplots()


# color = 'tab:red'
# ax1.set_xlabel('delta temperature (C)')
# ax1.set_ylabel('tau before dissasembly, s', color=color)
# ax1.scatter(x, data[:, 1]*0.2/0.8, label="len", color=color)
# ax1.plot(x, res_len.intercept + res_len.slope*x, color='pink', label='fitted len')
# ax1.tick_params(axis='y', labelcolor=color)
# ax1.set_xlim(20, 44)

# ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

# color = 'tab:blue'
# ax2.set_ylabel('speed of dissasembly, nm/s', color=color)  # we already handled the x-label with ax1
# ax2.scatter(x, slope*750, label="slope", color=color)
# # ax2.plot(x2, res_slope.intercept + res_slope.slope*x2, color='green', label='fitted speed')
# ax2.tick_params(axis='y', labelcolor=color)
# print("cap len slope", res_len.slope)
# # print("speed slope", res_slope.slope)

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# fig.legend()
# plt.show()