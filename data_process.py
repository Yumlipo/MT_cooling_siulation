from datetime import datetime as dt
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from screeninfo import get_monitors  # pip install screeninfo




# from get_img import mouse_action


def time_process(df, y):
    new_time = np.array([df[0]])
    new_ind = np.array([0])
    new_y = np.array([y[0]])
    for i in range(1, df.shape[0]):
        # print("time", df[i]-df[i-1], datetime.timedelta(0,1))
        if df[i]-df[i-1] > datetime.timedelta(0,1):
            while(df[i]-new_time[-1] > datetime.timedelta(0,1)):
                new_time = np.append(new_time, new_time[-1]+datetime.timedelta(0,1))
                # print('h', new_time[-1])
                new_ind = np.append(new_ind, new_ind[-1]+1)
                new_y = np.append(new_y, np.nan)
        new_time = np.append(new_time, df[i])
        new_ind = np.append(new_ind, new_ind[-1]+1)
        new_y = np.append(new_y, y[i])
    # print('new_ind', new_ind, df.shape[0])
    new_y = pd.Series(new_y).interpolate()
    return pd.Series(new_time), new_y


def angle_calc(coord):
    # coord = x11, y11, x12, y12, x21, y21, x22, y22
    m1 = (coord[3] - coord[1]) / (coord[2] - coord[0])
    m2 = (coord[7] - coord[5]) / (coord[6] - coord[4])
    tg_alpha = abs((m1-m2) / (1 + m1*m2))
    len1 = coord[2] - coord[0]
    len2 = coord[6] - coord[0]
    len_h1 = coord[3] - coord[1]
    len_h2 = coord[7] - coord[5]
    # print(m1, m2)
    
    return tg_alpha, m2, len1, len2, len_h1, len_h2


def temp_calc(len, arr_x, arr_y, start_ind):
    print("len and start", len, start_ind)
    temp_mean = np.mean(arr_y[start_ind:start_ind+len])
    return temp_mean

def show_graph(image, cooling_stack, t_array):
    # fig = plt.figure(figsize=(24, 16))
    drop_index = (np.where(cooling_stack[:-1] > cooling_stack[1:])[0] + 1)[0]
    print("drop", drop_index)
    monitor = get_monitors()[0]  # Primary monitor
    fig = plt.figure(figsize=((monitor.width-1000)/163, (monitor.height-1000)/163))
    fig.tight_layout()
    ax = []
    ax += [fig.add_subplot(211)]
    ax[0].imshow(image, aspect=0.05, interpolation="bilinear") # Display kymograph
    # xax = ax[0].axes.get_xaxis()
    # xax = xax.set_visible(False)
    ax[0].set_ylabel('MT lenght')
    ax[0].set_xlim(t_array[drop_index]-500, t_array[drop_index]+500)
    ax[0].axvline(x=t_array[drop_index], color='r', linestyle='--', linewidth=1)

    # t_array = t_array*100 #make axes like this so they share
    

    ax += [fig.add_subplot(212, sharex=ax[0])]
    
    # ax += [fig.add_subplot(212)]
    # ax[1].set_box_aspect(image.shape[0]/image.shape[1]*0.5)
    ax[1].scatter(t_array, cooling_stack, s=1)
    ax[1].set_xlabel('Time (s*100)')
    ax[1].set_ylabel('Temperature')
    ax[1].set_xlim(ax[0].get_xlim())  # Ensure exact match

    # start = np.where((t_array >= 4000))[0][0]
    ax[1].axvline(x=t_array[drop_index], color='r', linestyle='--', linewidth=1)
    
    fig.canvas.draw()
    img_plot = np.array(fig.canvas.renderer.buffer_rgba())
    img_inprogress = img_plot.copy()

    # cv2.waitKey(0) & 0xFF 

    return img_plot, img_inprogress