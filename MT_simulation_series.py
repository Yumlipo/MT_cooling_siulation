# MT_simulation_series
import numpy as np
import time
import MT_simulation

# Define the temperatures you want to test
# temperatures = [10, 2, -2, -6, -10]  # in °C
temperatures = [32]
# k_hydr_stack = [16, 17, 18, 19, 20, 21]
# k_hydr_stack = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
k_hydr_stack = [0.1]
iterations = 1

# Simulation parameters
time_of_calc = 60*5  # Total simulation time in seconds (2 minutes)
period = 10  # Cooling period (8s in time units)

c = 15

rates_32 = {
# Initial rate constants at reference temperature (32°C)
"k_off_d0": 650,      # Off rate for GDP-bound tubulin
"k_off_t0":  236,        # Off rate for GTP-bound tubulin
"k_hydr0": 0.4,  # Hydrolysis rate
"k_on_d0": 2 / 15 * c, # On GDP rate (polymerization rate) (маленькая, чтобы не было спасений)
"k_on_t0": 270 / 15 * c # On GTP rate (polymerization rate)
}

# Temperature coefficients for rate constants
rates_A = {
"A_off_d": -17,
"A_off_t": -5.3,
"A_hydr": 8,
"A_on_d": 4,
"A_on_t": 4
}

# Run simulation
start = time.time()
for k_h in k_hydr_stack:
    rates_32["k_hydr0"] = k_h
    # Calculate intercepts for linear temperature dependence
    rates_b = {
    "b_off_d": rates_32["k_off_d0"] - rates_A["A_off_d"]*32,
    "b_off_t": rates_32["k_off_t0"] - rates_A["A_off_t"]*32,
    "b_hydr": rates_32["k_hydr0"] - rates_A["A_hydr"]*32,
    "b_on_d": rates_32["k_on_d0"] - rates_A["A_on_d"]*32,
    "b_on_t": rates_32["k_on_t0"] - rates_A["A_on_d"]*32
    }
    print("b", rates_b["b_off_d"], rates_b["b_off_t"], rates_b["b_hydr"], rates_b["b_on_d"], rates_b["b_on_t"])
    for temp in temperatures:
        for i in range(iterations):
            print(f"\nRunning {i} simulation for T = {temp}°C and k_hydr = {k_h} 1/s")
            img, t_array, k, temp_func_calc, MT_cap, MT_length, max_len, mean_cap = MT_simulation.MT_calc(rates_A, rates_b, temp_goal=temp, time_of_calc=time_of_calc, period=period)

            print("Execution time:", time.time() - start)
            print("img", img.shape, t_array.shape)

            catastrophes, name = MT_simulation.visualize(img, t_array, k, temp_func_calc, MT_cap, MT_length, c, i, period, k_h)
            print("cat", catastrophes) 
            combined = np.column_stack((k_h, temp, max_len, mean_cap, catastrophes))
            # header='k_hydr/temperature/length/MT_cap_size/catastrophes'
            with open(name, 'a') as f:
                    # f.write('k_hydr/temperature/length/MT_cap_size/catastrophes\n')
                np.savetxt(f, combined, delimiter='\t')  # Appends to open file
            # try:
            #     img, t_array, k, temp_func_calc, MT_cap, MT_length, max_len = MT_simulation.MT_calc(rates_A, rates_b, temp_goal=temp, time_of_calc=time_of_calc, period=period)

            #     print("Execution time:", time.time() - start)
            #     print("img", img.shape, t_array.shape)

            #     catastrophes, name = MT_simulation.visualize(img, t_array, k, temp_func_calc, MT_cap, MT_length, c, i, period, k_h)
            #     print("cat", catastrophes) 
            #     combined = np.column_stack((k_h, temp, max_len, np.mean(MT_cap[~np.isnan(MT_cap)]), catastrophes))
            #     # header='k_hydr/temperature/length/MT_cap_size/catastrophes'
            #     with open(name, 'a') as f:
            #             # f.write('k_hydr/temperature/length/MT_cap_size/catastrophes\n')
            #         np.savetxt(f, combined, delimiter='\t')  # Appends to open file
            # except:
            #     print("can't make this simulation")
            
            # index += 1

# print("Len of MT", np.mean(len_MT), len_MT)
# print("Cap of MT", np.mean(MT_cap_stack), MT_cap_stack)
# print("catastrophes", np.mean(catastrophes), catastrophes)

