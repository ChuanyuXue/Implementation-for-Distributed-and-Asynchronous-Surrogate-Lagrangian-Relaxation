import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

f = open('DASLR_100.txt','r')
logs_all = eval(f.read().replace('array(','').replace(')',''))
f.close()

f = open('SLR_100.txt','r')
logs_all_2 = eval(f.read().replace('array(','').replace(')',''))
f.close()

logs_average_2 = dict()
logs_average_2_count = dict()
for i in logs_all_2:
    for time, value in i:
        logs_average_2.setdefault(time, np.zeros(2))
        logs_average_2_count.setdefault(time, 0)
        logs_average_2[time] += np.array(value)
        logs_average_2_count[time] += 1
for time in logs_average_2:
    logs_average_2[time] = logs_average_2[time] / logs_average_2_count[time]

logs_average_2 = [(key,logs_average_2[key]) for key in logs_average_2]


logs_average = dict()
logs_average_count = dict()
for i in logs_all:
    for time, value in i:
        logs_average.setdefault(time, np.zeros(2))
        logs_average_count.setdefault(time, 0)
        logs_average[time] += np.array(value)
        logs_average_count[time] += 1
for time in logs_average:
    logs_average[time] = logs_average[time] / logs_average_count[time]

logs_average = [(key,logs_average[key]) for key in logs_average]



optimal_lambda = np.array([0.6,0])
plt.figure(figsize=(20,8))

result = [np.linalg.norm(x[1] - optimal_lambda) for x in logs_average]
sns.lineplot([x[0] for x in logs_average][:1000],
            result[:1000], label='DASLR')

result = [np.linalg.norm(x[1] - optimal_lambda) for x in logs_average_2]
sns.lineplot([x[0] for x in logs_average_2][:1000],
            result[:1000], label='SLR')


plt.ylabel('Distance to the optimum')
plt.xlabel('Simulated time (sec)')
plt.yscale("log")
plt.savefig('result_in_100sec_100avg.pdf')