import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def filtered(arr, threshold):
        arr = np.array(arr)
        result = np.diff(arr)
        ind = np.where(result >= threshold)[0]
        print(ind, arr[ind])
        return arr[ind], ind

def filtered1(arr, threshold):
        filtered = [arr[0]]
        for v in arr[1:]:
            if abs(v - filtered[-1]) >= threshold:  
                filtered.append(v)
        return filtered


def get_cools(file_dir):
    cooling_stack = pd.read_excel(file_dir+'\cooling.xls', skiprows=lambda x: x % 2, usecols=[1, 3],  names=["Time", "Temperature"])

    cooling_stack["Time"] = pd.to_datetime(cooling_stack["Time"], format="%H:%M:%S")
    # print(cooling_stack.head)
    # print(cooling_stack.dtypes)

    ind = np.where(np.diff(cooling_stack['Temperature']) > 30)[0]
    # print("ind", ind, ind2)
    cooling_stack.loc[ind, 'Temperature'] = 31.0

    cooling_stack['diff'] = cooling_stack['Temperature'] - cooling_stack['Temperature'].shift(-1) > 2
    true_indexes = cooling_stack.index[cooling_stack['diff']].tolist()

    filtered_arr = filtered1(true_indexes, 40)

    # print(filtered)

    # plot = cooling_stack.plot(title="DataFrame Plot", x = 'Time', y = 'Temperature')
    # for i in filtered_arr:
    #     plt.axvline(x=cooling_stack["Time"][i], color='red', linestyle='--', linewidth=2)
    # plt.show()

    cools = []
    min_temp = []
    num = 0
    temperature = cooling_stack['Temperature'].to_list()
    #Все эти выделения нужны только для того, чтобы получить температуру во время охлаждений
    for i in filtered_arr:
        cools += [temperature[i+2:i+100]]
        # print("\n\n", cools[-1])
        min_temp += [int(min(cools[num]))]
        num += 1

    print(min_temp)

    return cooling_stack, min_temp

# get_cools('Y:\\Documents\\LAB\\MT_cooling\\dynamic\\2024.08.28_MT_dynamic_18uM_2421_tub')