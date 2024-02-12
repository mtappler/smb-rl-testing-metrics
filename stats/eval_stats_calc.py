import json
import glob
from collections import defaultdict
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

start_marker = "Reward:"

class Stats:
    def __init__(self,reward,overall_nc, avg_nc, avg_imp, avg_imp_win, n_wins):
        self.reward = reward
        self.overall_nc = overall_nc
        self.avg_nc = avg_nc
        self.avg_imp = avg_imp 
        self.avg_imp_win = avg_imp_win
        self.n_wins = n_wins


def read_eval_file(file_name):
    with open(file_name, "r") as fp:
        lines = fp.readlines()
        header = lines[0]
        #sut_names = json.loads(header.replace("(","[").replace(")","]"))
        start_indexes = [i for i, x in enumerate(lines) if start_marker in x]
        stats_list = []
        
        for i,start_index in enumerate(start_indexes):
            sut_i = start_index
            reward_line = lines[sut_i].replace("Reward:","").split("+-")
            reward = (float(reward_line[0]),float(reward_line[1]))

            overall_nc_line = lines[sut_i + 1].replace("Overall coverage:","").replace("(","[").replace(")","]")
            overall_nc_list = json.loads(overall_nc_line)
            overall_nc = overall_nc_list[0]
    
            avg_nc_line = lines[sut_i + 2].replace("Avg. trace cov.:","").split("+-")
            avg_nc = (float(avg_nc_line[0]),float(avg_nc_line[1]))
            
            avg_imp_line = lines[sut_i + 3].replace("Avg. Imp. :","").split("+-")
            avg_imp = (float(avg_imp_line[0]),float(avg_imp_line[1]))
                
            avg_imp_win_line = lines[sut_i + 4].replace("Avg. Imp. (win):","").split("+-")
            avg_imp_win = (float(avg_imp_win_line[0]),float(avg_imp_win_line[1]))
                
            n_win_line = lines[sut_i + 5].replace("# wins:","")
            n_wins = int(n_win_line)

            stats = Stats(reward, overall_nc, avg_nc, avg_imp,avg_imp_win,n_wins)
            stats_list.append(stats)
            
                
        return stats_list

def extract_sut_ep(sut_name):
     return int(sut_name.replace("SUTs/1-1/mario_net_","").replace("SUTs/1-4/mario_net_","").replace(".chkpt",""))
    
def create_nc_plot(stats_list, level,color):
    t = list(range(1000, 16000, 1000))

    overall_ncs = list(map(lambda s : s.overall_nc, stats_list))
    avg_ncs = list(map(lambda s : s.avg_nc[0], stats_list))

    avg_ncs_lower = list(map(lambda s : s.avg_nc[0] - s.avg_nc[1]/2, stats_list))
    avg_ncs_upper = list(map(lambda s : s.avg_nc[0] + s.avg_nc[1]/2, stats_list))

    plt.plot(t, overall_ncs, f'{color}--')
    plt.plot(t, avg_ncs, color)
    plt.fill_between(t,avg_ncs_lower,avg_ncs_upper,color=color,alpha=0.1)
    
  
    

def create_imp_plot(stats_list, level,color):
    t = list(range(1000, 16000, 1000))

    imps = list(map(lambda s : s.avg_imp[0], stats_list))
    
    imps_lower = list(map(lambda s : s.avg_imp[0] - s.avg_imp[1]/2, stats_list))
    imps_upper = list(map(lambda s : s.avg_imp[0] + s.avg_imp[1]/2, stats_list))
    imps_win = list(map(lambda s : s.avg_imp_win[0], stats_list))

    rewards = list(map(lambda s : s.reward[0], stats_list))
    #rewards_lower = list(map(lambda s : s.reward[0] - s.reward[0] - , stats_list))
    
    max_reward = max(rewards)
    max_imp = max(imps)
    rewards = list(map(lambda r : (r/max_reward) * max_imp, rewards))

    # red dashes, blue squares and green triangles
    plt.plot(t, imps, f'{color}')
    plt.plot(t, rewards, f'{color}--')
    plt.fill_between(t,imps_lower,imps_upper,color=color,alpha=0.1)
    
    #tikzplotlib.save(f"eval_nc_plot_{level}.tex")

def main():
    do_imp = False
    if do_imp:
        level = "1-4"
        file_name = f"../eval_results/eval_result_{level}.txt"
        stats_list = read_eval_file(file_name)
        create_imp_plot(stats_list, level,"b")
        level = "1-1"
        file_name = f"../eval_results/eval_result_{level}.txt"
        stats_list = read_eval_file(file_name)

        #create_nc_plot(stats_list, level)
        create_imp_plot(stats_list, level,"r")
        #plt.show()
        tikzplotlib.save("imp_plot.tex")
    else:    
        level = "1-4"
        file_name = f"../eval_results/eval_result_{level}.txt"
        stats_list = read_eval_file(file_name)
        create_nc_plot(stats_list, level,"b")
        level = "1-1"
        file_name = f"../eval_results/eval_result_{level}.txt"
        stats_list = read_eval_file(file_name)

        #create_nc_plot(stats_list, level)
        create_nc_plot(stats_list, level,"r")
        tikzplotlib.save("nc_plot.tex")
    
if __name__ == "__main__":
    main()
