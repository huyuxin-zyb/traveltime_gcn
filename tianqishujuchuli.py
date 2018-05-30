import pandas as pd
import matplotlib.pyplot as plt


reader=pd.read_csv('weather/guiyang.csv')
reader['date']=pd.to_datetime(reader['date'])
# reader.to_csv('weather/guiyang.csv',columns=['city','date','high_t','low_t','weather'],index=None)

reader=reader[reader['date']<='2017-5-31']
print(reader)
# plt.plot(reader['d_wind'],'.')
# plt.show()
plt.rcParams['font.sans-serif'] = ['SimHei']
sizes=[]
labels=['晴','小雨','中雨','大雨']
for key,value in reader.groupby('weather'):
    print(key,len(value))
    # labels.append(key)
    sizes.append(len(value))

explode = (0,0,0,0.1) #0.1表示将Hogs那一块凸显出来
plt.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
#startangle表示饼图的起始角度

plt.show()