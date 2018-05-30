import pickle as pkl
import networkx as nx
import random
from utils import *
from my_tool import *

graph_2 = {}


def recursive_sub(link_id,trip_l):
    # print link_id
    if link_id in trip_l:
        return
    trip_l.append(link_id)
    if link_id not in graph_2 :
        return
    ind=random.randint(0,len(graph_2[link_id])-1)
    return recursive_sub(graph_2[link_id][ind],trip_l)


def get_trip():
    link_dict = {}
    with open('raw/link_id_num.csv', 'r') as file_to_read:
        i = 0
        for lines in file_to_read:
            i += 1
            if i == 1: continue
            lines = lines.strip()
            line = lines.split(',')
            link_dict[line[1]] = line[0]
    with open('raw/gy_contest_link_top.txt', 'r') as file_to_read:
        i=0
        for lines in file_to_read:
            i+=1
            if i==1:continue
            lines=lines.strip()
            link_ID,in_links,out_links=lines.split(';')
            link_ID=int(link_dict[link_ID])
            if len(out_links)>0:
                graph_2[link_ID]=[int(link_dict[il]) for il in out_links.split('#')]

    link_inf={}
    with open('raw/gy_contest_link_info.txt', 'r') as file_to_read:
        i = 0
        for lines in file_to_read:
            i += 1
            if i == 1: continue
            lines = lines.strip()
            line = lines.split(',')
            link_id=int(link_dict[line[0]])
            link_inf[link_id] = float(line[1])
    ii=0
    fw = open('data/trip.txt', 'w')
    j=0
    for key,value in link_dict.items():
        trip_l=[]
        recursive_sub(int(value),trip_l)
        # print trip_l
        ss=0.0
        for i in range(min(5,len(trip_l))):
            ss+=link_inf[trip_l[i]]
        for i in range(6,len(trip_l)):
            ii+=1
            ss+=link_inf[trip_l[i]]
            fw.writelines(','.join([str(j),str(ii),str(i),str(ss),'#'.join([ str(ll) for ll in trip_l[:i]])]) + '\n')
        j+=1
    fw.close()
    # print ii


def get_graph():
    link_dict={}
    with open('raw/link_id_num.csv', 'r') as file_to_read:
        i=0
        for lines in file_to_read:
            i+=1
            if i==1:continue
            lines=lines.strip()
            line=lines.split(',')
            link_dict[line[1]]=line[0]
    graph_1={}
    graph_2={}
    graph_0={}
    with open('raw/gy_contest_link_top.txt', 'r') as file_to_read:
        i=0
        for lines in file_to_read:
            i+=1
            if i==1:continue
            lines=lines.strip()
            link_ID,in_links,out_links=lines.split(';')
            link_ID=int(link_dict[link_ID])
            graph_0[link_ID]=[]
            graph_1[link_ID]=[]
            graph_2[link_ID]=[]
            if len(in_links)>0:
                graph_1[link_ID]=[int(link_dict[il]) for il in in_links.split('#')]
            graph_0[link_ID].extend(graph_1[link_ID])
            if len(out_links)>0:
                graph_2[link_ID]=[int(link_dict[il]) for il in out_links.split('#')]
            graph_0[link_ID].extend(graph_2[link_ID])
    print graph_0
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph_1))
    print adj
    pkl.dump(graph_1, open('data/ind.guiyang_in.graph', "wb"))
    pkl.dump(graph_2, open('data/ind.guiyang_out.graph', "wb"))
    pkl.dump(graph_0, open('data/ind.guiyang.graph', "wb"))

def data_view(f_name='gy_contest_link_traveltime_training_data.txt'):
    f_name='raw/'+f_name
    date_dict=dict()
    with open(f_name, 'r') as file_to_read:
        i=0
        for lines in file_to_read:
            i+=1
            if i==1:continue
            lines=lines.strip()
            line=lines.split(',')
            if  not date_dict.has_key(line[1]):
                date_dict[line[1]]=0
            date_dict[line[1]]+=1
    date_dict=sorted(date_dict.items(),key=lambda item:item[0])
    print date_dict


def yichangzhi(arr):
    mean=np.mean(np.array(arr))
    std=np.std(np.array(arr))
    n_list=[]
    for a in arr:
        if a<=mean+2*std and a>=mean-2*std :
            n_list.append(a)
    return n_list


def fill_value(list1):
    list2 = [0 for _ in range(len(list1))]
    list3 = [1 for _ in range(len(list1))]
    for i in range(len(list1)):
        n_list = [li for li in list1[i] if li < 50]
        if len(n_list) > 2:
            n_list = yichangzhi(n_list)
        if len(n_list) > 0: list2[i] = float(np.mean(np.array(n_list)))
    s,c=0,0
    for i in range(len(list2)):
        if list2[i]!=0:
            s+=list2[i]
            c+=1
    for i in range(len(list2)):
        if list2[i] == 0:
            list3[i] = 0
            list2[i]=s/c
    return list2, list3


def his_graph(value,bin1,ind=11):
    if ind==0:
        return True
    if value.has_key(bin1 - ind):
        return his_graph(value,bin1,ind-1)
    return False


def get_sample(ex_list,f_name='quaterfinal_gy_cmp_training_traveltime.txt'):

    link_dict={}
    with open('raw/link_id_num.csv', 'r') as file_to_read:
        i=0
        for lines in file_to_read:
            i+=1
            if i==1:continue
            lines=lines.strip()
            line=lines.split(',')
            link_dict[line[1]]=line[0]
    link_inf = {}
    with open('raw/gy_contest_link_info.txt', 'r') as file_to_read:
        i = 0
        for lines in file_to_read:
            i += 1
            if i == 1: continue
            lines = lines.strip()
            line = lines.split(',')
            link_id=int(link_dict[line[0]])
            link_inf[link_id] = float(line[1])

    f_name = 'raw/' + f_name
    sample_dict = dict()
    with open(f_name, 'r') as file_to_read:
        i = 0
        for lines in file_to_read:
            i += 1
            if i == 1: continue
            lines=lines.strip()
            line=lines.split(',')
            if line[1] not in sample_dict:
                sample_dict[line[1]]={}
            bin1=int(line[2])
            if bin1/5 not in sample_dict[line[1]]:
                sample_dict[line[1]][bin1/5]=[[] for j in range(len(link_dict))]
            link_id=int(link_dict[line[0]])
            sample_dict[line[1]][bin1/5][link_id].append(link_inf[link_id]/float(line[3]))
    for key,value in sample_dict.items():
        mask={}
        if key not in ex_list and len(value)>36:
            print key,len(value)
            for bin1,feature in value.items():
                value[bin1],mask[bin1]=fill_value(feature)
            pkl.dump(value,open('data/beijing/ind.'+str(key)+'.sample','wb'))
            pkl.dump(mask,open('data/beijing_mask/ind.'+str(key)+'.mask','wb'))
            ex_list.append(key)
    return ex_list


if __name__ == "__main__":
    # get_graph()
    # get_trip()
    ex_list = []
    file_list=['gy_contest_link_traveltime_training_data.txt','gy_contest_traveltime_training_data_second.txt',
               'quaterfinal_gy_cmp_training_traveltime.txt']
    # get_sample(ex_list)
    for f in file_list:
        print 1
        ex_list=get_sample(ex_list,f)