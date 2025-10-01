import os, glob, time, re
import numpy as np
from PIL import Image
import cv2
from datetime import datetime as dt
import datetime
import nd2
from data_process import time_process
from data_process import angle_calc, temp_calc, show_graph

coord = np.array([])
start_point = []
angle = 0
slope2 = 0

def get_start(temp):
    dif = np.diff(temp) < -0.5
    print(dif, np.where(dif)[0])
    return np.where(dif)[0][0]

#draw lines and calculate parametrs 
def mouse_action(event,x,y,flags,param):
    global flag, coord, start_point, img_inprogress, image, angle, slope2, focuschange, cooling_stack, timestamp, start, cooling_focus, img_plot, cur_temp, num_of_file, dir_name, t_array, cooling_stack, f_name
    if event == cv2.EVENT_LBUTTONDOWN: #write down the beggining 
        flag = True
        start_point = [x, y]
        if coord.shape[0] in [0, 4]:
            coord = np.append(coord, start_point) #x1 y1 or x3 y3
    if event == cv2.EVENT_MOUSEMOVE and flag == True: #line in progress
        cv2.line(img_inprogress, start_point, [x, y], (0, 255, 0), 1)
        cv2.imshow("MT cooling", cv2.cvtColor(img_inprogress, cv2.COLOR_RGBA2BGR))
        img_inprogress = img_plot.copy()
    if event == cv2.EVENT_LBUTTONUP:# and len(start_point)>0:
        flag = False
        end_point = [x, y] 
        cv2.line(img_plot, start_point, end_point, (0, 255, 0), 1) 
        coord = np.append(coord, end_point) 
        print("start, end line", coord)
        if coord.shape[0] == 8: #if two lines drawn
            angle, slope2, len1, len2, len_h1, len_h2 = angle_calc(coord) #calcucale angle between lines, slope for second, len of both lines
            start_ind = get_start(cooling_stack)
            temp_mean = temp_calc(int(len1*0.8), t_array, cooling_stack, start_ind) #calculate mean temperature while MT cap is alive
            temp_mean2 = temp_calc(int(len2*0.8), t_array, cooling_stack, start_ind)
            print("we can get the angle", angle, slope2, temp_mean)
            # cv2.putText(img_plot, 'len1: ' + str(round(len1, 2)) + ', slope: ' + str(round(slope2, 2)) + ', mean cooling temperature ' + str(round(temp_mean, 2)), org, font, fontScale, color, thickness, cv2.LINE_AA)
            text = str(num_of_file) + ' len1: ' + str(round(len1, 2)) + ', slope: ' + str(round(slope2, 2)) + ', mean T1 ' + str(round(temp_mean, 1)) + ', mean T2 ' + str(round(temp_mean2, 1))
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 20)
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 1

            w = cv2.getWindowImageRect('MT cooling')[3]

            img_plot = cv2.putText(img_plot, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
            img_plot = cv2.putText(img_plot,  dir_name, (50, 40), font, 0.4, color, thickness, cv2.LINE_AA)
            # img_plot = cv2.putText(img_plot,  str(round(temp_mean2, 1)), (50, 60), font, 0.3, color, thickness, cv2.LINE_AA)
            
            cv2.line(img_plot, [int(coord[0]), 50], [int(coord[0]), w-20], (255, 0, 0), 1) 
            cv2.line(img_plot, [int(coord[2]), 50], [int(coord[2]), w-20], (255, 0, 0), 1) 
            cv2.line(img_plot, [int(coord[6]), 50], [int(coord[6]), w-20], (255, 0, 0), 1) 
            # cv2.line(img_plot, [int(focuschange[0]*2), 70], [int(focuschange[0]*2+len1), w-50], (0, 255, 0), 1) 

            # img_plot, img_inprogress = show_graph(image, cooling_stack, timestamp, start, cooling_focus, arr_x, arr_y, [focuschange[0], focuschange[0]+int(len1), focuschange[0]+int(len2)], text, mean_val=temp_mean)
            with open("Y:\\Documents\\Python\\MT_simulation_cooling\\results.txt", 'a') as res_f:
                res_f.write(str(num_of_file) + " " + str(len1) + " " + str(len2) + " "  + str(len_h1) + " " + str(len_h2) + " " + str(temp_mean) + " " + str(temp_mean2) + " " + str(cur_temp) + " " + dir_name + "\n")
                # res_f.write(str(num_of_file) + " " + str(len1) + " " + str(slope2) + " " + str(temp_mean) + " " + str(cur_temp) + " " + dir_name + "\n")
            
            try:
                cv2.imwrite(dir_name+'\\'+str(num_of_file)+"_"+f_name[:-4]+"_saved_graph.png",  cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR))
            except:
                print("we couldn't save the pic: ", dir_name+'\\' + str(num_of_file))
            
            print("ratio", image.shape[1]/len1)
            # img_plot, img_inprogress = show_graph(image, cooling_stack, timestamp, start, cooling_focus, arr_x, arr_y, lines=[focuschange[0], focuschange[0]+int(len1*0.8), focuschange[0]+int((len1+len2)*0.8)])

            coord = np.array([])
        if coord.shape[0] > 8:
            coord = np.array([])


        start_point = []
        cv2.imshow("MT cooling", cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR))

num_of_file = 0


# for filename in glob.glob("Y:\\Documents\\LAB\\MT_cooling\\dynamic\\*\\*\\*(RGB).png", recursive=True):
for filename in glob.glob("Y:\\Documents\\LAB\\MT_cooling\\simulation\\2025.06.05_MT_simulation_1_15uM_10s_cooling\\*\\kymo*.png", recursive=True):
    print("Open now", filename)
    num_of_file += 1
    try:
        with open(filename, 'r') as f:
            cur_temp = 0
            image = []
            
            dir_name = os.path.dirname(filename)
            f_name = os.path.split(filename)[1]
            dir = os.path.split(os.path.dirname(filename))
            dir2 = os.path.split(os.path.dirname(dir[0]))
            print(dir_name+"\\cools.txt")
            prefix = f_name[4:-4]
            print("name", f_name[4:-4])

            try:
                cooling_stack = np.loadtxt(dir_name+"\\cools" + prefix + ".txt", delimiter=',')
                t_array = np.loadtxt(dir_name+"\\time"+prefix + ".txt", delimiter=',')
            except:
                print("can't load txt")


            # cooling_stack, min_temp = get_cools(dir[0])
            # min_temp = [10, 2, -2, -6, -10]
            # cooling_stack['Time'], cooling_stack['Temperature'] = time_process(cooling_stack['Time'], cooling_stack['Temperature']) #дополняем пропущенные значения
            # print(cooling_stack.head)

            # exact_time = dt.now().strftime('%X')
            # cooling_focus = np.array([])
            # timestamp = np.array([])

            # filename_nd2 = re.match(r'.*.nd2_', dir[1])
            # print(filename_nd2.group(0)[:-1])
            # f_nd2_name = dir[0] + '\\' + filename_nd2.group(0)[:-1]
            # f_nd2_name = dir[0] + '\\' + dir[1][:-1]
            # print("_", f_nd2_name, "_")
            # f_nd2_name = "Y:\\Documents\\LAB\\MT_cooling\\dynamic\\2024.07.26_MT_dynamic_14uM_2421_Tub\\5_14uN_tub_dynamic_cooling_-6_again.nd2"
            # with nd2.ND2File(f_nd2_name) as f_meta:
                # exact_time = dt.strptime(f_meta.text_info['date'][-8:].strip(), '%H:%M:%S')
                # print("time from file", exact_time.strftime('%X'))
                # for data in f_meta.events():
                    # timestamp = np.append(timestamp, data["Time [s]"])
                    # cooling_focus = np.append(cooling_focus, data['Ti ZDrive [µm]'])
                    # print(timestamp, cooling_focus)

            # with open(r"Y:/Documents/LAB/MT_cooling/dynamic/" + str(dir2[1]) + "/Metadata.txt", 'r') as fp:
            #     # read all lines using readline()
            #     lines = fp.readlines()
            #     for row in lines:
            #         # check if string present on a current line
            #         word = 'TextInfoItem_914.08.202'
            #         word2 = 'PFS Offset'
            #         word3 = 'timestamp'

            #         if row.find(word) != -1:
            #             print('string exists in file')
            #             ttime = row[-12:-1].strip()
            #             exact_time = dt.strptime(row[-12:-1].strip(), '%H:%M:%S')
            #             print("time from file", exact_time.strftime('%X'))
            #         if row.find(word2) != -1:
            #             # print(row[-7:-3])
            #             cooling_focus = np.append(cooling_focus, int(row[-7:-3]))
            #         if row.find(word3) != -1:
            #             tmp = list(filter(lambda x: (x in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']), row[-14:-3]))
            #             tmp = ''.join(tmp)
            #             if float(tmp) - timestamp[-1] > 10:
            #                 tmp = tmp[1:-1]
            #             timestamp = np.append(timestamp, float(tmp))

            # timestamp = timestamp[1:-1]
            # _, focuschange = filtered(cooling_focus, 2)
            # focuschange = focuschange - 1
            # print("filtered focus", focuschange)

            im = np.rot90(Image.open(f.name))
            # im = im.convert('RGB')
            image = np.array(im)
            # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            # image = image[:, :cooling_focus.shape[0], :].copy()
            print("Img shape", image.shape)
            img_inprogress = image.copy()

            # data = data.transpose()

            #вынимаем из названия температуру охлаждения
            m = re.search('_(..)C', dir[1][1:-1])
            print("where temp", dir[1][1:-1])
            if m:
                cur_temp = m.group(1)
            if int(cur_temp) == 66:
                cur_temp = '-10'
            print("temp ", int(cur_temp))
            cur_temp = int(cur_temp)

            # try:
            #     index = min_temp.index(cur_temp)
            # except:
            #     try:
            #         index = min_temp.index(cur_temp-1)
            #     except:
            #         index =  min_temp.index(cur_temp+1)

            # result = cooling_stack[cooling_stack['Time'] == exact_time]
            # print(result)
            # start = result.index.to_list()[0]
            # print(start)  

            # result = cooling_stack[cooling_stack['Time'] == dt.strptime('17:37:02', '%H:%M:%S')]
            # stop = result.index.to_list()[0]
            # print(stop) 

            # print("shape", cooling_stack.shape)
            # start1 = start + int(timestamp[0])
            
            # stop = start1 + image.shape[1] + 50
            # x = np.array(cooling_stack['Time'])
            # y = np.array(cooling_stack['Temperature'])

            # if stop < x.shape[0]:
            #     print("here is an error", x, start, start1)
            #     x = x[start1:stop]
            #     y = y[start1:stop]
            #     print("start", start, start1, x[0])
            #     print("x shape", x.shape)
            #     print("time finished", x[-1])

            #     arr_y = np.array([y[0]])
            #     arr_x = np.array([x[0]])
            #     for i in range(1, y.shape[0]):
            #         for r in range(4):
            #             arr_y = np.append(arr_y, y[i-1]+(y[i]-y[i-1]) / 5 * (r+1))
            #             arr_x = np.append(arr_x, x[i-1]+(x[i]-x[i-1]) / 5 * (r+1))
            #         arr_y = np.append(arr_y, y[i])
            #         arr_x = np.append(arr_x, x[i])
            #     print("gr size", arr_x.shape, arr_y.shape)

            img_plot, img_inprogress = show_graph(image, cooling_stack, t_array)

            flag = False
            # start = get_start(cooling_stack)
            

            cv2.namedWindow('MT cooling', cv2.WINDOW_FULLSCREEN)
            w = cv2.getWindowImageRect('MT cooling')[3]
            # cv2.line(img_plot, [start, 50], [start, w-20], (255, 0, 0), 1)
            cv2.imshow('MT cooling', cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR))
            cv2.setMouseCallback('MT cooling', mouse_action)
            cv2.waitKey(0) & 0xFF #press q to close current window and go to the next one

                
                # if key == ord("q"):
                #     cv2.destroyAllWindows()
                # cv2.destroyAllWindows()
            
                # cv2.setMouseCallback('MT cooling', mouse_action)


                    
                    # return image, img_plot
    except:
        print("Can't open ", filename)