import numpy as np
import random
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import exp
import time
from scipy.optimize import curve_fit
from scipy import interpolate
from datetime import date
today = date.today()
import os
from PIL import Image
from catastrophes import catastrophe_calc



def exponential_decay(x, A, lambd):
    return A * np.exp(-lambd * x)

def k_calc(b, A, T):
    """Calculate temperature-dependent rate constant using linear model"""
    # Ensure rate is never negative (minimum value is 0.000001)

    # k_b = 1.38*10**(-23)
    # return k*np.exp(-A*(32-T)/((278+32)*(278+T)))
    return np.maximum(0.000001, b+A*T)
    # return -k*A*(32-T)

def rand_MT(k):
    """Generate random time for next event using exponential distribution"""

    r = random.random()
    return -np.log(r)/k

def temperature(temp_goal, time_of_calc, period):
    """Generate temperature profile over time"""
    time_oc = time_of_calc*100
    per = period*100
    temp_func = np.zeros(time_oc+1)  # High resolution temperature array
    start_cooling = (time_oc - 6000)   # When cooling starts

    # Initial phase at 32°C
    temp_func[:start_cooling] = 32

    # Linear cooling phase
    num = (32-temp_goal)/per
    for i in range(per):
        temp_func[start_cooling+i] = temp_func[start_cooling+i-1] - num

    # Final phase at target temperature
    temp_func[(start_cooling+per):] = temp_goal
    # temp_func[(start_cooling+period):(start_cooling+period+1000)] = temp_goal
    # for i in range(period):
    #     temp_func[(start_cooling+period+1000)+i] = temp_func[(start_cooling+period+1000)+i-1] + num
    # temp_func[(start_cooling+2*period+1000):] = 32

    # Plot temperature profile
    # plt.plot(np.linspace(0, time_of_calc+1, time_oc+1), temp_func)
    # plt.xlabel("time, s")
    # plt.ylabel('temperature')
    # plt.show()
    return temp_func, start_cooling

# # Plot rate constants vs temperature
# t_x = np.arange(-14, 34, 2)
# plt.plot(t_x, k_calc(b_off_t, A_off_t, t_x), label="k_off_t", color="blue")
# plt.plot(t_x, k_calc(b_off_d, A_off_d, t_x), label="k_off_d", color="green")
# plt.plot(t_x, k_calc(b_hydr, A_hydr, t_x), label="k_hydr", color="orange")
# plt.plot(t_x, k_calc(b_on, A_on, t_x), label="k_on", color="red")
# plt.xlabel("temperature")
# plt.ylabel('k')
# plt.title('rates')
# plt.legend()
# plt.grid(color='0.95')
# plt.show()

def MT_cap_calculation_1(MT):
    """Calculate MT cap size"""
    MT = MT[::-1]
    # my_hist, bin_edges = np.histogram(MT, bins=40)
    # print(my_hist, bin_edges)
    gtf = np.zeros(MT.shape[0])
    gtf = np.where(MT == 2., gtf, 1)
    gdf = np.zeros(MT.shape[0])
    gdf = np.where(MT == 1., gdf, 1)
    # print(gdf)

    # Get indices where value == 2
    positions = np.where(MT == 2)[0]

    #get hist
    hist, bin_edges = np.histogram(positions, bins=len(MT)//20, range=(0, len(MT)), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of bins

    # Fit bins 
    # significant_bins = hist > 0.01 * np.max(hist)  # Keep bins with >1% of max density
    try:
        params, _ = curve_fit(exponential_decay, bin_centers, hist, p0=[0.0001, 0.00001])
    except:
        params = [0, 0]

    A, lambd = params

    print(f"Fit: P(x) = {A:.5f} * exp(-{lambd:.6f} x)")

    epsilon = 0.5  # Define "significant" as 1% of max probability
    # x_cutoff = -np.log(epsilon) / lambd if lambd > 0 else len(MT)
    x_cutoff = 1 / lambd if lambd > 0 else len(MT)
    significant_positions = positions[positions <= x_cutoff]

    print(f"Significant positions x ≤ {x_cutoff:.1f}")
    # print(positions)

    cap_size = significant_positions[-1]
    
    fig = plt.figure()
    ax = []
    ax += [fig.add_subplot(211)]
    ax[0].bar(
        np.arange(0, MT.shape[0]), 
        gtf, 
        width=1, 
        color="green",
        edgecolor='none'
    ) 
    ax[0].bar(
        np.arange(0, MT.shape[0]), 
        gdf, 
        width=1, 
        color="orange",
        edgecolor='none'
    ) 
    ax[0].axvline(x=significant_positions[-1], color='r', linestyle='--', label='Threshold')

    # Plot histogram
    ax += [fig.add_subplot(212)]

    ax[1].bar(bin_centers, hist, width=20, alpha=0.5)
    ax[1].scatter(bin_centers, hist)
    # ax[1].bar(bin_centers, hist, width=20, alpha=0.3)
    # ax[1].hist(significant_positions, bins=len(MT)//20, range=(0, len(MT)), alpha=0.3)
    ax[1].axvline(x=significant_positions[-1], color='r', linestyle='--', label='Threshold')
    ax[1].plot(bin_centers, exponential_decay(bin_centers, A, lambd), 'r-', label=f"Fit: {A:.5f} exp(-{lambd:.6f} x)")
    ax[1].set_xlabel('Position along array')
    ax[1].set_ylabel('Frequency of 2')
    plt.legend()
    plt.show()

def MT_cap_calculation(MT):
    """Calculate MT cap size"""
    MT = MT[::-1]
    # my_hist, bin_edges = np.histogram(MT, bins=40)
    # print(my_hist, bin_edges)
    gtf = np.zeros(MT.shape[0])
    gtf = np.where(MT == 2., gtf, 1)
    gdf = np.zeros(MT.shape[0])
    gdf = np.where(MT == 1., gdf, 1)


    # Get indices where value == 2
    positions = np.where(MT == 2)[0]

    #get hist
    hist, bin_edges = np.histogram(positions, bins=len(MT)//20, range=(0, len(MT)), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of bins

    # Fit bins 
    # significant_bins = hist > 0.01 * np.max(hist)  # Keep bins with >1% of max density
    params, _ = curve_fit(exponential_decay, bin_centers, hist, p0=[0.0001, 0.00001])

    A, lambd = params

    epsilon = 0.5  # Define "significant" as 1% of max probability
    # x_cutoff = -np.log(epsilon) / lambd if lambd > 0 else len(MT)
    x_cutoff = 1 / lambd if lambd > 0 else len(MT)
    significant_positions = positions[positions <= x_cutoff]

    # print(len(significant_positions), significant_positions)

    return significant_positions[-1], len(positions), len(significant_positions)




def MT_calc(rates_A, rates_b, temp_goal=32, time_of_calc=60, period=8):
    """Main microtubule simulation function"""
    t = 0  # Current time
    index = 0  # Current index in arrays
    index_kymo = 0
    ind_len = 0  # Current microtubule length

    predicted_size = time_of_calc*100+1  # Initial array size
    max_MT_size = 1000  # Initial microtubule size
    max_len_actual = 0
    
    # Initialize arrays
    t_array = np.zeros(predicted_size)  # Time points
    temp_func_calc = np.zeros(predicted_size)
    MT = np.zeros(max_MT_size)          # Microtubule state array
    MT_cap = np.full(predicted_size, np.nan)   # Microtubule cap state
    MT_length = np.zeros(predicted_size)

    kymo = np.zeros((predicted_size, max_MT_size))  # Kymograph array

    temp_func, start_cooling = temperature(32, time_of_calc, period)  # Get temperature profile
    cooling_flag = 0 
    mean_cap = 0

    # Rate constant arrays
    k = {
        "k_off_d": np.zeros(predicted_size),
        "k_off_t": np.zeros(predicted_size),
        "k_hydr": np.zeros(predicted_size),
        "k_on_d": np.zeros(predicted_size),
        "k_on_t": np.zeros(predicted_size)
    }

    print("before", kymo.shape)

    index_kymo_prev = 0
    
    # Main simulation loop
    while t < time_of_calc:
        if t < time_of_calc - 60 and ind_len < 2000 and cooling_flag == 0:
            temp = 32
        else:
            if cooling_flag == 0:
                start_cooling = index_kymo
                mean_cap = np.nanmean(MT_cap[index-100:index])
                print(mean_cap)
            cooling_flag = 1
            temp = temp_goal
            
    # temp = temp_func[index_kymo] # Get current temperature
        
        temp_func_calc[index] = temp

        # Calculate current rate constants
        
        cap_positions = np.where(MT == 2)[0]
        n = len(cap_positions)
        # print("n", cap_positions, n, ind_len)
        # print("n", n)
        k["k_off_d"][index] = k_calc(rates_b["b_off_d"], rates_A["A_off_d"], temp)
        k["k_off_t"][index] = k_calc(rates_b["b_off_t"], rates_A["A_off_t"], temp)
        k["k_hydr"][index] = k_calc(rates_b["b_hydr"], rates_A["A_hydr"],  temp) * n
        k["k_on_d"][index] = k_calc(rates_b["b_on_d"], rates_A["A_on_d"], temp)
        k["k_on_t"][index] = k_calc(rates_b["b_on_t"], rates_A["A_on_t"], temp)
        # print("k ", k["k_off_d"][index], k["k_off_t"][index], k["k_hydr"][index], k["k_on"][index])

        # Generate random times for possible events
        suffix_off = "d" if ind_len > 2 and MT[ind_len-2] == 1 else "t"
        suffix_on = "d" if ind_len > 1 and MT[ind_len-1] == 1 else "t"
        const = [
            rand_MT(k[f"k_off_{suffix_off}"][index]),
            rand_MT(k["k_hydr"][index]),
            rand_MT(k[f"k_on_{suffix_on}"][index])
        ]
        # print(suffix)

        # Find which event happens now
        const_num = list(enumerate(const, 0))        
        time_min = min(const_num, key=lambda i: i[1])
        # print("time_min", time_min)
        t += time_min[1] # Advance overall time
        # if t > 60 and t < 63:
        #     print(t)
        #     print(index)
        #     print("i", int(t*100), temp)
        t_array[index] = t

        event = time_min[0] # Which event occurred
        # if index < 10:
        #     print("const", const_num)
        #     print("k ", k["k_off_d"][index], k["k_off_t"][index], k["k_hydr"][index], k["k_on"][index])
        #     print("event", time_min[0], time_min[1])
        #     print("ind", index, ind_len, MT[:ind_len])

        kymo[index_kymo_prev:index_kymo] = MT[None, :]

        # Handle events
        if event == 0: #shortening (depolymerization)
            # print("shortening")
            if ind_len > 0:
                ind_len -= 1
                MT[ind_len] = event
        elif event == 1: #hydrolysis (GTP → GDP)
            # print("hydro")
            GTP_index = np.where(MT == 2)[0] # Find all GTP-bound subunits
            if GTP_index.shape[0] > 0:
                i = np.random.choice(GTP_index, size=1, replace=False)
                MT[i] = event # Convert to GDP-bound
        else: #growning (polymerization)
            # print("grow")
            MT[ind_len] = event
            ind_len += 1
        MT_length[index] = ind_len

        

        # Record current state
        
        # if index in [1000, 2000, 3000, 4000, 5000] and ind_len>100:
        #     print(ind_len)
        #     MT_cap_calculation_1(MT[:ind_len])
        # if index%100 == 0:
        #     try:
        #         # print("len", ind_len)
        #         i1, i2, i3 = MT_cap_calculation(MT[:ind_len])
        #         MT_cap[index] = i1
        #         # print(MT_cap[index])
        #     except:
        #         MT_cap[index] = np.nan
        try:
            MT_cap[index] = ind_len-cap_positions[0]
        except:
            MT_cap[index] = 0
        index_kymo_prev = index_kymo
        index_kymo = round(t*100)
        index += 1

        # Expand arrays if microtubule is getting too long
        try:
            if ind_len > MT.shape[0]-1:
                kymo = np.concatenate((kymo, np.zeros((kymo.shape[0], max_MT_size))), axis=1)
                MT = np.concatenate((MT, np.zeros(max_MT_size)))
                print("add", MT.shape, kymo.shape)
        except:
            print("can't extend lenght arrays")
            break
        
        try:
            if index > t_array.shape[0]-1:
                kymo = np.concatenate((kymo, np.zeros((predicted_size, kymo.shape[1]))), axis=0)
                t_array = np.concatenate((t_array, np.zeros(predicted_size)))
                MT_cap = np.concatenate((MT_cap, np.zeros(predicted_size)))
                MT_length = np.concatenate((MT_length, np.zeros(predicted_size)))

                k = {
                    "k_off_d": np.concatenate((k["k_off_d"], np.zeros(predicted_size))),
                    "k_off_t": np.concatenate((k["k_off_t"], np.zeros(predicted_size))),
                    "k_hydr": np.concatenate((k["k_hydr"], np.zeros(predicted_size))),
                    "k_on_d": np.concatenate((k["k_on_d"], np.zeros(predicted_size))),
                    "k_on_t": np.concatenate((k["k_on_t"], np.zeros(predicted_size)))
                }
                temp_func_calc = np.concatenate((temp_func_calc, np.zeros(predicted_size)))
                print("add time", kymo.shape, t_array.shape)
        except:
            print("can't extend time arrays")
            index_kymo_prev = index_kymo
            break
        
        # Expand time arrays if time is continued

        # print("index", index)
        if ind_len > max_len_actual:
            max_len_actual = ind_len
    kymo[index_kymo_prev] = MT

    print("after", index, index_kymo, max_len_actual)
    print(kymo.shape, t_array.shape, MT_cap.shape, k["k_on_d"].shape)
    
    if temp_goal < 15:
        print("start cooling", start_cooling)
        last_non_zero_row = start_cooling + 1000
        try:
            time_ind = np.where((t_array >= last_non_zero_row/100-0.001))[0][0] + 1
        except:
            print("except 2")
            time_ind = index
        print("last non zero", last_non_zero_row, time_ind, t_array[time_ind])
        print(start_cooling+period)    
    else:
        last_non_zero_row = index_kymo
        time_ind = index      

    # Trim arrays to actual size used
    k["k_off_d"] = k["k_off_d"][:time_ind]
    k["k_off_t"] = k["k_off_t"][:time_ind]
    k["k_hydr"] = k["k_hydr"][:time_ind]
    k["k_on_d"] = k["k_on_d"][:time_ind]
    k["k_on_t"] = k["k_on_t"][:time_ind]
    
    # k["k_off_d"] = k["k_off_d"][::100]
    # k["k_off_t"] = k["k_off_t"][::100]
    # k["k_hydr"] = k["k_hydr"][::100]
    # k["k_on"] = k["k_on"][::100]
    

    kymo = kymo[:last_non_zero_row]
    # kymo = kymo[::100]
    t_array = t_array[:time_ind]
    # t_array = t_array[::100]
    temp_func_calc = temp_func_calc[:time_ind]
    # temp_func_calc = temp_func_calc[::100]
    MT_cap = MT_cap[:time_ind]
    # MT_cap = MT_cap[::100]
    MT_length = MT_length[:time_ind]

    print(kymo.shape, t_array.shape, MT_cap.shape, k["k_on_d"].shape)

    print(t_array, np.mean(np.diff(t_array)))
    # MT_cap = kymo_cap_calc(kymo, predicted_size)        
    return kymo, t_array, k, temp_func_calc, MT_cap, MT_length, max_len_actual, mean_cap



def visualize(img, t_array, k, temp_func_calc, MT_cap, MT_length, c, i, period, k_h):
    print("\n=== DEBUG INFO ===")
    print(f"Shapes - img: {img.shape}, t_array: {t_array.shape}")
    print(f"MT_cap NaN count: {np.isnan(MT_cap).sum()}/{len(MT_cap)}")
    print(f"Temperature range: {temp_func_calc.min():.1f} to {temp_func_calc.max():.1f}°C")
    img_ori = img.copy()
    # Prepare data for visualization
    img = np.rot90(img)  # Rotate kymograph for proper orientation

    # fig = plt.figure()
    # fig.canvas.manager.full_screen_toggle() 
    # fig.tight_layout()
    fig = plt.figure(figsize=(16, 12))
    ax = []
    ax += [fig.add_subplot(321)]
    ax[0].imshow(img, aspect=0.5, interpolation="bilinear") # Display kymograph
    # xax = ax[0].axes.get_xaxis()
    # xax = xax.set_visible(False)
    ax[0].set_ylabel('MT length')

    t_array = t_array*100 #make axes like this so they share

    # Plot rate constants over time
    x= np.linspace(0, t_array.shape[0]//100, t_array.shape[0])
    ax += [fig.add_subplot(323, sharex=ax[0])]
    # ax[1].set_box_aspect(img.shape[0]/img.shape[1]*0.5)
    ax[1].scatter(t_array, k["k_off_t"], label="k_off_t", color="blue", s=1)
    ax[1].scatter(t_array, k["k_hydr"], label="k_hydr", color="orange", s=1)
    ax[1].scatter(t_array, k["k_on_d"], label="k_on_d", color="red", s=1)
    ax[1].scatter(t_array, k["k_on_t"], label="k_on_t", color="pink", s=1)
    ax[1].scatter(t_array, k["k_off_d"], label="k_off_d", color="green", s=1)
    ax[1].set_xlim(0, t_array[-1])
    ax[1].set_ylabel('k')
    ax[1].legend()

    # Plot GDP off-rate separately
    # ax += [fig.add_subplot(324, sharex=ax[0])]
    # # ax[2].set_box_aspect(img.shape[0]/img.shape[1]*0.5)
    # ax[2].scatter(t_array, k["k_off_d"], label="k_off_d", color="green", s=1)
    # ax[2].set_xlim(0, t_array[-1])
    # ax[2].set_ylabel('k')
    # ax[2].set_xlabel('Time (s*100)')
    # ax[2].legend()

    # Plot temperature profile
    ax += [fig.add_subplot(325, sharex=ax[0])]
    # ax[3].set_box_aspect(img.shape[0]/img.shape[1]*0.5)
    ax[-1].scatter(t_array, temp_func_calc, s=1)
    ax[-1].set_xlabel('Time (s*100)')
    ax[-1].set_ylabel('Temperature')
    ax[-1].set_xlim(0, t_array[-1])

    #Plot MT cap size
    ax += [fig.add_subplot(322, sharex=ax[0])]
    # ax[4].set_box_aspect(img.shape[0]/img.shape[1]*0.5)
    valid_idx = ~np.isnan(MT_cap) #& (MT_cap > 5)
    ax[-1].scatter(t_array[valid_idx], MT_cap[valid_idx], s=1)
    ax[-1].set_xlabel('Time (s*100)')
    ax[-1].set_ylabel('MT cap size')
    ax[-1].set_xlim(0, t_array[-1])

    #Plot MT length and catastrophes
    ax += [fig.add_subplot(324, sharex=ax[0])]
    ax[-1].scatter(t_array, MT_length, s=1)
    ax[-1].set_xlabel('Time (s*100)')
    ax[-1].set_ylabel('MT length')
    ax[-1].set_xlim(0, t_array[-1])
    

    # print("After graph time:", time.time() - start)
    # plt.savefig('fig'+str(k_off_d0)+','+str(k_off_t0)+','+str(k_hydr0)+','+str(k_on0)+','+str(temp_goal)+'.png')
    t_of_exp = str(round(temp_func_calc[-1]))
    if t_of_exp == '-10':
        t_of_exp = '66'
    if t_of_exp in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
        t_of_exp = '+' + t_of_exp
    
    dir = "Y:\\Documents\\LAB\\MT_cooling\\simulation\\" + today.strftime("%Y.%m.%d")+"_MT_simulation_1_" + str(c) +"uM_" + str(period)+"s_cooling\\_" + t_of_exp + "C\\"
    
    try:
        os.makedirs(dir, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directory {dir}: {e}")
        return
    
    new_i=0
    fig_name = dir + 'fig_exp' + str(new_i) +"_" + str(i) + '_kh_' + str(k_h)+'.png'
    cat_name = dir + 'cat_exp' + str(new_i) +"_" + str(i) + '_kh_' + str(k_h)+'.png'
    kymo_name = dir + 'kymo_exp' + str(new_i) +"_" + str(i) + '_kh_' + str(k_h)+'.png'
    time_name = dir + 'time_exp' + str(new_i) +"_" + str(i) + '_kh_' + str(k_h)+'.txt'
    cools_name = dir + 'cools_exp' + str(new_i) +"_" + str(i) + '_kh_' + str(k_h)+'.txt'
    results_name = dir + "temp_len_cap_cat" + ".txt"
    
    
    while os.path.exists(fig_name) or os.path.exists(kymo_name):
        new_i += 1
        fig_name = dir+'fig_exp' + str(new_i) +"_" + str(i)  + '_kh_' + str(k_h)+'.png'
        kymo_name = dir+'kymo_exp' + str(new_i) +"_" + str(i)  + '_kh_' + str(k_h)+'.png'
        cat_name = dir+'cat_exp' + str(new_i) +"_" + str(i)  + '_kh_' + str(k_h)+'.png'
        time_name = dir+'time_exp' + str(new_i) +"_" + str(i)  + '_kh_' + str(k_h)+'.txt'
        cools_name = dir+'cools_exp' + str(new_i) +"_" + str(i)  + '_kh_' + str(k_h)+'.txt'
    # while os.path.exists(cat_name):
    #     ii += 1
    #     results_name = dir+"temp_len_cap_cat" + str(ii) + '.txt'
    print(fig_name, kymo_name, cat_name)    
    
    # plt.show()
    try:
        plt.savefig(fig_name)
        print("saved img success")
    except:
        print("can't save this picture")
    
    try:
        cat = catastrophe_calc(MT_length, cat_name)
        print("saved cat success")
    except:
        cat = 0
        print("can't calculate catastrophe")

    # plt.imshow(img)
    # plt.savefig(kymo_name)

    # For float arrays (0-1)
    normalized = (img_ori - img_ori.min()) / (img_ori.max() - img_ori.min())
    Image.fromarray((normalized * 255).astype('uint8')).save(kymo_name)
    

    try:
        np.savetxt(cools_name, temp_func_calc, delimiter=',')
        np.savetxt(time_name, t_array, delimiter=',')
        print("saved txt success")
    except:
        print("can't save txts")
    plt.close('all') 
    # plt.show()

    # plt.scatter(x, t_array, s=1)
    # plt.show()

    # cooling_stack = make_cools()
    return cat, results_name
