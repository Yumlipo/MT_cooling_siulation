import os,glob
import numpy as np
import re
import matplotlib.pyplot as plt
import scipy.stats as stat
import matplotlib as mpl
import matplotlib as mpl
from scipy.optimize import curve_fit


def line(x, a, b):
   return a*x+b
#Initial parametrs
sec_frame  = 20
nm_px = 102

#lists for full data
mean = []
angle = []
lenght = []
rate_full = []

name_dir = []
name_file = []

#index of opened directory
i=0

#dictionary for results
#store data divided by types of MT.
# store temperatures that were in experiments in the order that files were opened.
#store indexes of corresponding files
#store calculated rates (velocity)
results = {'temp': [],
        'indexes': [],
        'rate': [],
        'e': [],
        'len_v': []}


#calculating shortening rate
#Считаем скорость для каждой кимограммы и потом вычисляем среднее
def v_calc(ang, temp):
   v = - (nm_px * 60)/(np.tan(np.pi * ang/ 180)*1000*sec_frame)
   # print("v_calc, ang ", ang)
   print('v = ', round(np.mean(v), 4))
   print('STD = ', round(np.std(v), 4))
   print('SEM = ', round(stat.sem(v), 4))
   print('len v ', len(v))
   return np.mean(v), v, round(np.std(v), 4)

def calculate_all(res, dir, rate_full, cur_temp, data, i):
   temp = 'temp'
   index = 'indexes'
   rate = 'rate'
   ee = 'e'
   lv = 'len_v'
   
   # добавляем новые значения в словарь
   # tmp = results_my[temp]
   # tmp += [int(cur_temp)]
   res[temp] += [int(cur_temp)]

   # tmp = results_my[index]
   # tmp += [i]
   res[index] += [i]

   # tmp = results_my[rate]
   v_mean, vv, e = v_calc(data, int(cur_temp))
   rate_full += [vv]
   # tmp += [v_mean]#вычисленная скорость
   res[rate] += [v_mean]

   res[ee] += [e]

   res[lv] += [len(vv)]
   return res, rate_full

#searching all results_my in all experiments
#папка с данными со всех эксп
# folder/*/*/Results.csv
# "Y:\\Documents\\LAB\\MT_cooling\\simulation\\2025.05.19_MT_simulation_1_15uM_8s_cooling\\_32C\\1temp_len_cap_cat.txt"
for filename in glob.glob("Y:\\Documents\\LAB\\MT_cooling\\simulation\\2025.06.05_MT_simulation_1_15uM_10s_cooling\\*\\Results.csv", recursive=True):
# for filename in glob.glob("C:\\Users\\YummyPolly\\Documents\\LAB\\MT_cooling\\*\\*\\Results.csv", recursive=True):
   with open(filename, 'r') as f:
      data = np.loadtxt(f, delimiter=',', skiprows=1, usecols = (2) )
      # data = data.transpose()
      dir = os.path.split(os.path.dirname(filename))

      print('\n', dir[1])
      print("dir", os.path.basename(dir[0]))
      # print(data)

      #вынимаем из названия температуру охлаждения. Не понятно, что делать с ростом.
      # m = re.search('_(..)C', dir[1])
      # if m:
      #    cur_temp = m.group(1)
      # if int(cur_temp) == 66:
      #    cur_temp = '-10'
      # print("temp ", int(cur_temp))

      cur_temp = 0


      #temperature += [int(found)]

      name_dir += [os.path.basename(dir[0])]
      name_file += [dir[1]]
      # print(name_dir)

      #на всякий случай записываем все данные
      # mean += [data[1, :]]
      # angle += [data[2, :]]
      # lenght += [data[3, :]]

      #делаем разделение по дням.
    #   if name_dir[i] == '2024.03.14_GMPCPP_1.5uM_7' or name_dir[i] == '2024.03.20_GMPCPP_1.5uM_9':
    #      results_what, rate_full = calculate_all(results_what, dir, rate_full, cur_temp, data, i)
    #      print('what')
    #   elif name_dir[i] == '2024.02.14_GMPCPP_1.5uM_3' or name_dir[i] == '2024.03.08_Taxol_3' or name_dir[i] == '2024.03.19_GMPCPP_1.5uM_8':
    #      results_il, rate_full = calculate_all(results_il, dir, rate_full, cur_temp, data, i)
    #      print('il')
    #   else:
    #      results_my, rate_full = calculate_all(results_my, dir, rate_full, cur_temp, data, i)
    #      print('my')
      results, rate_full = calculate_all(results, dir, rate_full, cur_temp, data, i)

      i += 1#идем открывать следующий файл

# print(len(mean))
# print(results_my)
# print(*angle, sep='\n')



#plotting results_my
fig, ax2 = plt.subplots()
# fig.set_size_inches(13.5, 10.5)
# fig.tight_layout()


plt.rcParams['font.size'] = '22'
# csfont = {'fontname':'Calibri Light'}
print(mpl.font_manager.get_font_names())
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Set tick font size
for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
    label.set_fontsize(16)


ax2.scatter(results['temp'], results['rate'], linewidth=4.0, color="red", label="rate")

av_temp_GMPCPP = np.array([-8, -6, -4, -2, 0, 2, 10, 15, 32])
av_v_GMPCPP = np.array([-0.4348,-0.330683333,-0.2789,-0.125325,-0.104266667,-0.0238,0.0016,0.0124,0.192552778])
av_err_GMPCPP = np.array([0.267536627,0.132700156,0.1301183,0.180307701,0.15361942,0.150115467,0.15,0.15,0.188232183])

# ax2.set_title('GMPCPP')
ax2.set_xlabel('temperature, °C', loc='right', fontsize='18')
ax2.set_ylabel('disassembly rate, nm/s', fontsize='18')

ax2.set_xlim(-12, 12)
# ax1.set_ylim(-1, 0.75)
ax2.grid(True)
ax2.set_xticks(np.arange(-12, 12, 2))

temp_np = np.array(results['temp'])
results_np = np.array(results['rate'])
try:
   params, _ = curve_fit(line, temp_np, results_np, p0=[-1, 1])
except:
   params = [0, 0]
a, b = params

print(f"Fit: y(x) = {a:.5f} * x + {b:.6f}")
ax2.plot(temp_np, line(temp_np, a, b), label="estimated slope")

ax2.legend()
print("rate", results['rate'])
plt.show()



with open("v.txt", "w") as f:
   f.write(str(rate_full))
   f.write(str(name_dir))

def print_full(results, ff):
    i_gmp = 0
    i_t = 0
    for i in results['indexes']:
        ff.write(str(name_dir[i]) + ',')
        ff.write(str(name_file[i]) + ',')
        ff.write(str(results['temp'][i_gmp]) + ',')
        ff.write(str(round(results['rate'][i_gmp], 4)) + ',')
        ff.write(str(results['e'][i_gmp]) + ',')
        ff.write(str(results['len_v'][i_gmp]) + '\n')
        i_gmp += 1

with open("output.txt", "w") as ff:
   ff.write("dir, file, temp,  <v>, STD, len v \n")
   print_full(results, ff)

# with open("output/" + "power.txt", "wb") as f:
#    np.savetxt(f, power)

def print_v(results, ff):
    for i in range(len(results['temp'])):
        ff.write(str(results['temp'][i]) + ',')
        ff.write(str(round(results['rate'][i], 4)) + ',')
        ff.write(str(results['e'][i]) + '\n')
    

with open("rate.txt", "w") as ff:
   ff.write("temp,  <v>, STD\n")
   print_v(results, ff)
