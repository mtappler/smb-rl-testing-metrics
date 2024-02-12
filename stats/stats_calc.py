from dataclasses import dataclass
import json
import glob
from scipy import stats
from collections import defaultdict
from eval_stats_calc import read_eval_file
import numpy as np

import matplotlib.pyplot as plt
import tikzplotlib

class CoverageSummary:
    def __init__(self,coverage_line):
        cov_list_line = coverage_line.replace("(","[").replace(")","]")
        cov_list = json.loads(cov_list_line)
        self.full_cov = cov_list[0]
        self.layer_cov = cov_list[1]
@dataclass
class TestSuiteStats:
    pos_to_tc_stats : dict
    
    def compute_avg_uncertainty(self):
        return sum([s.uncertainty for s in self.pos_to_tc_stats.values()]) / len(self.pos_to_tc_stats)
    
    def compute_avg_imps(self):
        avg_imp = 0
        for s in self.pos_to_tc_stats.values():
            avg_imp += sum([tcs.imp_value for tcs in s.sut_stats]) / len(s.sut_stats)
            
        return avg_imp / len(self.pos_to_tc_stats)

    def compute_avg_agg_ncs(self):
        return self.get_uncert_and_avg_agg_nc_at_end()[1]
        
    def collect_uncert_and_avg_curr_nc_values(self):
        uncert_and_curr_nc = []
        for tc_stats in self.pos_to_tc_stats.values():
            sut_stats = tc_stats.sut_stats
            avg_curr_nc = sum([s.curr_nc.full_cov for s in sut_stats]) / len(sut_stats)
            uncert = tc_stats.uncertainty
            uncert_and_curr_nc.append((uncert,avg_curr_nc))
        return uncert_and_curr_nc
    
    def collect_uncert_and_avg_imps(self):
        uncert_and_imp = []
        for tc_stats in self.pos_to_tc_stats.values():
            sut_stats = tc_stats.sut_stats
            avg_imp = sum([s.imp_value for s in sut_stats]) / len(sut_stats)
            uncert = tc_stats.uncertainty
            uncert_and_imp.append((uncert,avg_imp))
        return uncert_and_imp
    
    def get_uncert_and_avg_agg_nc_at_end(self):
        avg_uncert = self.compute_avg_uncertainty()
        max_pos = max(self.pos_to_tc_stats.keys())
        sut_stats = self.pos_to_tc_stats[max_pos].sut_stats
        avg_agg_nc = sum([s.agg_nc.full_cov for s in sut_stats]) / len(sut_stats)
        return (avg_uncert,avg_agg_nc)

    def collect_train_eps_and_agg_nc(self):
        max_pos = max(self.pos_to_tc_stats.keys())
        sut_stats = self.pos_to_tc_stats[max_pos].sut_stats
        train_eps_and_agg_nc = [(s.train_ep,s.agg_nc.full_cov) for s in sut_stats]
        return train_eps_and_agg_nc   
    
class TestCaseSutStats:
    sut_prefix = "mario_net_"
    sut_suffix = ".chkpt"
    def __init__(self,sut_name : str,agg_nc : CoverageSummary,curr_nc : CoverageSummary,imp_value : float):
        self.sut_name = sut_name
        self.agg_nc = agg_nc
        self.curr_nc = curr_nc
        self.imp_value = imp_value
        start_ep = sut_name.index(TestCaseSutStats.sut_prefix) + len(TestCaseSutStats.sut_prefix)
        end_ep = sut_name.index(TestCaseSutStats.sut_suffix)
        self.train_ep = int(sut_name[start_ep:end_ep])

@dataclass
class TestCaseStats:
    def __init__(self,test_x_pos : int, sut_stats : list, conclusive : bool, difference : float):
        self.test_x_pos = test_x_pos
        self.sut_stats = sut_stats
        self.sut_stats.sort(key = lambda sut_stat : extract_sut_ep(sut_stat.sut_name))
        self.conclusive = conclusive
        self.difference = difference
        self.uncertainty = 1 - ((difference - 8/15) / (7/15)) if conclusive else 1


def extract_sut_ep(sut_name):
     return int(sut_name.replace("SUTs/1-1/mario_net_","").replace("SUTs/1-4/mario_net_","").replace(".chkpt",""))

def index_where(l : list, start : int, predicate):
    while start < len(l):
        if predicate(l[start]):
            return start
        start += 1
    return -1 
    
    
def read_single_file(file_name):
    with open(file_name, "r") as fp:
        lines = fp.readlines()
        test_start_indexes = [i for i, x in enumerate(lines) if "Tested x-coordinate" in x]
        pos_to_tc_stats = dict()
        for i,start_index in enumerate(test_start_indexes):
            x_pos_line = lines[start_index]
            test_x_pos = int(x_pos_line.replace("Tested x-coordinate:","").strip())
            sut_i = start_index + 1
            suts_end = index_where(lines, sut_i, lambda line: "Conclusive" in line)
            sut_stats = []
            while sut_i < suts_end:
                sut_name = lines[sut_i].replace(":","").strip()
                agg_nc = CoverageSummary(lines[sut_i + 1])
                curr_nc = CoverageSummary(lines[sut_i + 2])
                imp_value = float(lines[sut_i + 3])
                sut_i += 4
                tc_sut_stats = TestCaseSutStats(sut_name,agg_nc,curr_nc,imp_value)
                sut_stats.append(tc_sut_stats)
            conc_line = lines[sut_i]
            uncert_line = lines[sut_i + 1]
            conc = "True" in conc_line
            if not conc:
                print("INCONC:",file_name)
            diff_val = float(uncert_line.replace("Difference:","").strip())
            tc_stats = TestCaseStats(test_x_pos,sut_stats,conc,diff_val)
            pos_to_tc_stats[test_x_pos] = tc_stats
        ts_stats = TestSuiteStats(pos_to_tc_stats)
        return ts_stats

def read_files(file_prefix):
    stats_dict = dict()
    for file_name in glob.glob(file_prefix):
        stats_dict[file_name] = read_single_file(file_name)
        if "_rand" in file_name:
            stats_dict[file_name].test_type = "random"
        elif "_bp" in file_name:
            stats_dict[file_name].test_type = "boundary_point"
        elif "_eqd" in file_name:
            stats_dict[file_name].test_type = "equi_dist"
    return stats_dict

def collect_all_train_and_agg_ncs(stats_dict):
    train_eps_and_agg_nc = []
    for ts_stats in stats_dict.values():
        train_eps_and_agg_nc.extend(ts_stats.collect_train_eps_and_agg_nc())
    return train_eps_and_agg_nc

def corr_train_ep_agg_nc(stats_dict):
    train_ep_and_agg_nc = collect_all_train_and_agg_ncs(stats_dict)
    train_eps,agg_ncs = zip(*train_ep_and_agg_nc)
    pearson_coeff, _ = stats.pearsonr(train_eps,agg_ncs)
    spearman_coeff, _ = stats.spearmanr(train_eps,agg_ncs)
    print("Train eps vs. agg nc")
    print(f"Pearson coeff: {pearson_coeff}")
    print(f"Spearman coeff: {spearman_coeff}")

def corr_uncert_avg_curr_nc(stats_dict):
    uncert_and_curr_nc = collect_all_uncert_and_avg_curr_nc_values(stats_dict)
    uncert_vals,curr_ncs = zip(*uncert_and_curr_nc)
    pearson_coeff, _ = stats.pearsonr(uncert_vals,curr_ncs)
    spearman_coeff, _ = stats.spearmanr(uncert_vals,curr_ncs)
    print("Uncertainty vs. curr nc")
    print(f"Pearson coeff: {pearson_coeff}")
    print(f"Spearman coeff: {spearman_coeff}")

def corr_uncert_avg_imp(stats_dict):
    uncert_and_imp = collect_all_uncert_and_avg_imp_values(stats_dict)
    uncert_vals,imps = zip(*uncert_and_imp)
    pearson_coeff, _ = stats.pearsonr(uncert_vals,imps)
    spearman_coeff, _ = stats.spearmanr(uncert_vals,imps)
    print("Uncertainty vs. imps")
    print(f"Pearson coeff: {pearson_coeff}")
    print(f"Spearman coeff: {spearman_coeff}")
    
def corr_uncert_agg_nc_at_end(stats_dict):
    uncert_and_agg_nc = collect_all_uncert_and_avg_agg_nc_at_end(stats_dict)
    uncert_vals,agg_ncs = zip(*uncert_and_agg_nc)
    pearson_coeff, _ = stats.pearsonr(uncert_vals,agg_ncs)
    spearman_coeff, _ = stats.spearmanr(uncert_vals,agg_ncs)
    print("Avg. uncertainty vs. agg nc")
    print(f"Pearson coeff: {pearson_coeff}")
    print(f"Spearman coeff: {spearman_coeff}")
    
def collect_all_uncert_and_avg_agg_nc_at_end(stats_dict):
    uncert_and_agg_nc = []
    for ts_stats in stats_dict.values():
        uncert_and_agg_nc.append(ts_stats.get_uncert_and_avg_agg_nc_at_end())
    return uncert_and_agg_nc

def collect_all_uncert_and_avg_imp_values(stats_dict):
    uncert_and_imp = []
    for ts_stats in stats_dict.values():
        uncert_and_imp.extend(ts_stats.collect_uncert_and_avg_imps())
    return uncert_and_imp
    
def collect_all_uncert_and_avg_curr_nc_values(stats_dict):
    uncert_and_agg_nc = []
    for ts_stats in stats_dict.values():
        uncert_and_agg_nc.extend(ts_stats.collect_uncert_and_avg_curr_nc_values())
    return uncert_and_agg_nc

# uncertainty values are stupidly named
def compute_avg_uncertainty(stats_dict):
    print("Avg. uncertaintys")
    stats_list = []
    for ts_name,ts_stats in stats_dict.items():
        stats_list.append((ts_name,ts_stats.compute_avg_uncertainty()))
    stats_list.sort(key = lambda x: x[1])
    for ts_name,uncert in stats_list:
        print(f"{ts_name}: {uncert}")

        # uncertainty values are stupidly named
def compute_avg_importance(stats_dict):
    print("Avg. uncertaintys")
    stats_list = []
    for ts_name,ts_stats in stats_dict.items():
        stats_list.append((ts_name,ts_stats.compute_avg_imps()))
    stats_list.sort(key = lambda x: x[1])
    for ts_name,uncert in stats_list:
        print(f"{ts_name}: {uncert}")


        # uncertainty values are stupidly named
def compute_avg_uncert_importance(stats_dict):
    print("Avg. uncertaintys")
    stats_list = []
    for ts_name,ts_stats in stats_dict.items():
        stats_list.append((ts_name,ts_stats.compute_avg_imps() + ts_stats.compute_avg_uncertainty()))
    stats_list.sort(key = lambda x: x[1])
    for ts_name,uncert in stats_list:
        print(f"{ts_name}: {uncert}")

def compute_avg_agg_nc(stats_dict):
    print("Avg. agg NCs")
    stats_list = []
    for ts_name,ts_stats in stats_dict.items():
        stats_list.append((ts_name,ts_stats.compute_avg_agg_ncs()))
    stats_list.sort(key = lambda x: x[1])
    overall_avg_nc = 0
    for ts_name,agg_nc in stats_list:
        print(f"{ts_name}: {agg_nc}")
        overall_avg_nc += agg_nc
    _,only_agg_nc = zip(*stats_list)
    print(f"Overall average: {overall_avg_nc/len(stats_list)}")
    print(f"Min: {min(only_agg_nc)}")
    print(f"Max: {max(only_agg_nc)}")

def print_sut_tt_stats(avg_results, header, description):
    sut_tts = sorted(list(avg_results.keys()))
    for (tt,sut) in sut_tts:
        print(header)
        print(f"Test type: {tt}, SUT: {sut}")
        print(f"{description}: {avg_results[(tt,sut)]}")

def avg_imp_value_per_sut_and_tt(stats):
    imp_vals = defaultdict(list)
    for ts_stat in stats.values():
        for tc_stat in ts_stat.pos_to_tc_stats.values():
            for sut_stat in tc_stat.sut_stats:
                imp_vals[(ts_stat.test_type,sut_stat.sut_name)].append(sut_stat.imp_value)
    return {sut_tt : sum(imp_list)/len(imp_list)  for sut_tt,imp_list in imp_vals.items()} 
        
def split_into_t_type(stats_dict):
    stats_dict_list = defaultdict(list)
    for stat in stats_dict.values():
        stats_dict_list[stat.test_type].append(stat)
    return stats_dict_list

def compute_avg_agg_nc_evo(stats_list):
    avg_agg_nc = np.zeros(15)
    for stat in stats_list:
        max_pos = max(stat.pos_to_tc_stats.keys())
        tc_stat_at_max = stat.pos_to_tc_stats[max_pos]
        agg_nc_array = np.array(list(map(lambda sut_stat: sut_stat.agg_nc.full_cov,tc_stat_at_max.sut_stats)))
        avg_agg_nc = avg_agg_nc + agg_nc_array
    return avg_agg_nc / len(stats_list)

def avg_imps_in_test_suite(stat):
    avg_imp = np.zeros(15)
    for tc_stat in stat.pos_to_tc_stats.values():
        sut_stats = tc_stat.sut_stats
        imp_values_at_tc = np.array(list(map(lambda tcs_stat: tcs_stat.imp_value,sut_stats)))
        avg_imp += imp_values_at_tc
    return avg_imp / len(stat.pos_to_tc_stats)

def compute_avg_imp_evo(stats_list):
    avg_imp = np.zeros(15)
    for stat in stats_list:
        agg_imps_in_stat = avg_imps_in_test_suite(stat)
        avg_imp = avg_imp + agg_imps_in_stat
    return avg_imp / len(stats_list)

def compute_avg_agg_nc_evo_per_tt(stats_dl):
    evo_dict = {tt : compute_avg_agg_nc_evo(stats_list) for tt, stats_list in stats_dl.items()}
    return evo_dict

def compute_avg_imp_evo_per_tt(stats_dl):
    evo_dict = {tt : compute_avg_imp_evo(stats_list) for tt, stats_list in stats_dl.items()}
    return evo_dict

def tt_for_paper(tt):
    if "boundary" in tt:
        return "BP"
    elif "rand" in tt:
        return "RAND"
    elif "equi" in tt:
        return "EQD"

def plot_value_evolution_per_tt(evo_dict,tt_color_pairs, eval_stats_list,level,eval_selector,marker="",show=True):
    t = list(range(1000, 16000, 1000))
    eval_values = list(map(eval_selector, eval_stats_list))
    plt.plot(t, eval_values, "k--",alpha=0.5)
    
    for (tt,color) in tt_color_pairs:
        values = evo_dict[tt] 
        plt.plot(t, values, f"{color}{marker}",label=f"{tt_for_paper(tt)}({level})")
        #plt.fill_between(t,avg_ncs_lower,avg_ncs_upper,color=color,alpha=0.1)
    if show:
        plt.show()

def compute_avg_uncert(stats_list):
    avg_uncert = 0
    for ts_stats in stats_list:
        avg_uncert += ts_stats.compute_avg_uncertainty()
    return avg_uncert / len(stats_list)

tt_color_pairs = [("random","r"),("boundary_point","k"),("equi_dist","b")]
analysis = "uncert"

if analysis == "nc-plot":
    levels_markers = [("1-1","o-"), ("1-4","^-")]
    for (level,marker) in levels_markers:
        file_name = f"../eval_results/eval_result_{level}.txt"
        eval_stats_list = read_eval_file(file_name)
        file_prefix = f"../coverage_results/test_result_{level}_*"
        stats_dict = read_files(file_prefix)
        stats_dict_list = split_into_t_type(stats_dict)
        evo_dict = compute_avg_agg_nc_evo_per_tt(stats_dict_list)
        plot_value_evolution_per_tt(evo_dict,tt_color_pairs,eval_stats_list,level,
                                    lambda s : s.overall_nc, marker=marker,
                                    show=False)
    plt.legend()
    #plt.show()
    
    tikzplotlib.save("nc_testing_vs_eval.tex")

if analysis == "imp-plot":
    levels_markers = [("1-1","o-"), ("1-4","^-")]
    for (level,marker) in levels_markers:
        file_name = f"../eval_results/eval_result_{level}.txt"
        eval_stats_list = read_eval_file(file_name)
        file_prefix = f"../coverage_results/test_result_{level}_*"
        stats_dict = read_files(file_prefix)
        stats_dict_list = split_into_t_type(stats_dict)
        evo_dict = compute_avg_imp_evo_per_tt(stats_dict_list)
        plot_value_evolution_per_tt(evo_dict,tt_color_pairs,eval_stats_list,level,
                                    lambda s : s.avg_imp[0], marker=marker,
                                    show=False)
    plt.legend()
    #plt.show()
    tikzplotlib.save("imp_testing_vs_eval.tex")
if analysis == "uncert":
    levels_markers = [("1-1","o-"), ("1-4","^-")]
    for (level,marker) in levels_markers:
        file_name = f"../eval_results/eval_result_{level}.txt"
        eval_stats_list = read_eval_file(file_name)
        file_prefix = f"../coverage_results/test_result_{level}_*"
        stats_dict = read_files(file_prefix)
        stats_dict_list = split_into_t_type(stats_dict)
        avg_uncert_per_tt = {tt : compute_avg_uncert(stats_list) for tt, stats_list in stats_dict_list.items()}
        print("Avg. uncertainty per test type")
        for tt in avg_uncert_per_tt.keys():
            print(f"{tt}: {avg_uncert_per_tt[tt]}")

    #tikzplotlib.save("imp_testing_vs_eval.tex")
else:
    file_prefix = f"../coverage_results/test_result_*"
    stats_dict = read_files(file_prefix)
    
    
    corr_uncert_avg_curr_nc(stats_dict)
    corr_uncert_avg_imp(stats_dict)
    corr_uncert_agg_nc_at_end(stats_dict)
    corr_train_ep_agg_nc(stats_dict)
    #compute_avg_agg_nc(stats_dict)
    
    #print_sut_tt_stats(avg_imp_value_per_sut_and_tt(stats_dict), "Importance values:", "Imporance")
    #compute_avg_uncert_importance(stats_dict)
