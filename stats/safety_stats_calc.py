import json
import glob
from collections import defaultdict
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

start_tc_marker = "Tested x-coordinate:"


class TcSutStat:
    def __init__(self,sut_name,safety_ratio, conc_tests):
        self.sut_name = sut_name
        self.safety_ratio = safety_ratio
        self.conc_tests = conc_tests

class TcStat:
    def __init__(self,test_x_pos,sut_stats):
        self.test_x_pos = test_x_pos
        self.sut_stats = sut_stats

class TsStat:
    def __init__(self,pos_to_tc_stats):
        self.pos_to_tc_stats = pos_to_tc_stats
        self.test_type = "unknown"

def index_where(l : list, start : int, predicate):
    while start < len(l):
        if predicate(l[start]):
            return start
        start += 1
    return -1

def read_file(file_name):
    with open(file_name, "r") as fp:
        lines = fp.readlines()
        header = lines[0]
        #sut_names = json.loads(header.replace("(","[").replace(")","]"))
        test_start_indexes = [i for i, x in enumerate(lines) if "Tested x-coordinate" in x]
        pos_to_tc_stats = dict()
        for i,start_index in enumerate(test_start_indexes):
            x_pos_line = lines[start_index]
            test_x_pos = int(x_pos_line.replace("Tested x-coordinate:","").strip())
            sut_i = start_index + 1
            suts_end = index_where(lines, sut_i, lambda line: start_tc_marker in line or not line.strip())
            sut_stats = []
            while sut_i < suts_end:
                sut_name = lines[sut_i].replace(":","").strip()
                safety_ratio = float(lines[sut_i + 1].replace("Safety ratio:",""))
                conc_tests = int(lines[sut_i + 2].replace("Conclusive tests:", ""))
                sut_i += 3
                tc_sut_stats = TcSutStat(sut_name,safety_ratio,conc_tests)
                sut_stats.append(tc_sut_stats)
            tc_stats = TcStat(test_x_pos,sut_stats)
            pos_to_tc_stats[test_x_pos] = tc_stats
    ts_stats = TsStat(pos_to_tc_stats)
    return ts_stats

def read_files(file_prefix):
    stats_dict = dict()
    for file_name in glob.glob(file_prefix):
        print(file_name)
        stats_dict[file_name] = read_file(file_name)
        if "_rand" in file_name:
            stats_dict[file_name].test_type = "random"
        elif "_bp" in file_name:
            stats_dict[file_name].test_type = "boundary_point"
        elif "_eqd" in file_name:
            stats_dict[file_name].test_type = "equi_dist"
   
    return stats_dict

def avg_ratio_per_sut_and_tt(stats):
    ratios = defaultdict(list)
    for ts_stat in stats.values():
        for tc_stat in ts_stat.pos_to_tc_stats.values():
            for sut_stat in tc_stat.sut_stats:
                ratios[(ts_stat.test_type,sut_stat.sut_name)].append(sut_stat.safety_ratio)
    return {sut_tt : sum(ratio_list)/len(ratio_list)  for sut_tt,ratio_list in ratios.items()} 
    
def print_sut_tt_stats(avg_results, header, description):
    sut_tts = sorted(list(avg_results.keys()))
    for (tt,sut) in sut_tts:
        print(header)
        print(f"Test type: {tt}, SUT: {sut}")
        print(f"{description}: {avg_results[(tt,sut)]}")

def extract_sut_ep(sut_name):
     return int(sut_name.replace("SUTs/1-1/mario_net_","").replace("SUTs/1-4/mario_net_","").replace(".chkpt",""))
    
def create_bar_plot(avg_results):
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    sut_tts = sorted(list(avg_results.keys()), key = lambda x : (x[0],extract_sut_ep(x[1]) ))
    safety_ratio_means = defaultdict(list)
    # group by ts type, thus first in list must be BP for example
    # dict items must be suts
    tt_types = []
    for (tt,sut) in sut_tts:
        safety_ratio_means[extract_sut_ep(sut)].append(round(avg_results[(tt,sut)],2))
        if tt not in tt_types:
            tt_types.append(tt)
    # tuple them
    safety_ratio_means = {k : tuple(v) for k,v in safety_ratio_means.items()}
    tt_types = tuple(map(lambda tt: "BP" if "boundary" in tt else "EQD" if "equi" in tt else "RAND", tt_types))
    
    x = np.arange(len(tt_types))  # the label locations
        
    for attribute, measurement in safety_ratio_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Safety Ratio')
    ax.set_title('Safety Ratios by Test-Case Selection')
    ax.set_xticks(x + width, tt_types)
    ax.legend(loc='upper left')
    ax.set_ylim(0.6, 1.0)
    plt.show()
    #tikzplotlib.save("safety_ratio_1-4.tex")
file_name_prefix = "../safety_ratios_500_ep/test_result_1-1_extra_*"
stats = read_files(file_name_prefix)
avg_ratios = avg_ratio_per_sut_and_tt(stats)

print_sut_tt_stats(avg_ratios, "Fail ratios:", "Safety ratio")


create_bar_plot(avg_ratios)
