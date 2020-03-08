import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
import os
from tkinter import *
from pandas import ExcelWriter
plt.rcParams.update({'font.size': 16})
plt.rcParams["figure.figsize"] = (15,10)
import sys
sys.path.append("/Users/anagtv/Desktop/Cyclotron_python")
sys.path.append("/Users/anagtv/Documents/Beta-Beat.src-master")
from tfs_files import tfs_pandas
from mpl_interaction import figure_pz
import matplotlib.pyplot as plt


COLORS = ['#1E90FF','#FF4500','#32CD32',"#6A5ACD","#20B2AA","#00008B","#A52A2A","#228B22"]

def _parse_args():
    parser = OptionParser()
    parser.add_option("-i", "--input",
                    help="Input measurement path",
                    metavar="INPUT", dest="input_path")
    parser.add_option("-o", "--output",
                    help="Output measurement path",
                    metavar="OUTPUT", dest="output_path")
    parser.add_option("-c", "--current",
                    help="Target current",
                    metavar="TCURRENT", dest="target_current")
    options, _ = parser.parse_args()
    return options.input_path,options.output_path,options.target_current

def get_data_tuple(path_file):
    all_parts = []
    logfile = open(path_file,"r")
    print ("path")
    print (path_file)
    for line in logfile:
         parts = line.split()
         all_parts.append(
            parts)
    target_number = (np.array(all_parts[0])[1])
    real_values = all_parts[4:]
    return real_values,target_number 

def get_data(real_values):
    print ("real values here")
    #print (real_values[0])
    data_df = pd.DataFrame.from_records(real_values)
    column_names = ["Time","Arc_I","Arc_V","Gas_flow","Dee_1_kV","Dee_2_kV","Magnet_I","Foil_I","Coll_l_I","Target_I","Coll_r_I","Vacuum_P","Target_P","Delta_Dee_kV","Phase_load","Dee_ref_V","Probe_I","He_cool_P","Flap1_pos","Flap2_pos","Step_pos","Extr_pos","Balance","RF_fwd_W","RF_refl_W","Foil_No"]
    data_df = data_df.drop([0,1,2], axis=0)
    data_df.columns = column_names
    return data_df

def get_time(excel_data_df,current):
    time = excel_data_df.Time[excel_data_df['Target_I'].astype(float) > float(current)]
    relative_time = time 
    return time

def get_collimator_parameters(excel_data_df,current):
    collimator_r = excel_data_df.Coll_r_I[excel_data_df['Target_I'].astype(float) > float(current)].astype(float)
    collimator_l = excel_data_df.Coll_l_I[excel_data_df['Target_I'].astype(float) > float(current)].astype(float)
    return collimator_r,collimator_l

def get_source_parameters(excel_data_df,current):
    source_voltage = excel_data_df.Arc_V[excel_data_df['Target_I'].astype(float) > float(current)].astype(float)
    gas_flow = excel_data_df.Gas_flow[excel_data_df['Target_I'].astype(float) > float(current)].astype(float)
    source_current = excel_data_df.Arc_I[excel_data_df['Target_I'].astype(float) > float(current)].astype(float)
    return source_voltage,source_current,gas_flow

def get_rf_parameters(excel_data_df,current):
    dee2_voltage = excel_data_df.Dee_2_kV[excel_data_df['Target_I'].astype(float) > float(current)].astype(float)
    dee1_voltage = excel_data_df.Dee_1_kV[excel_data_df['Target_I'].astype(float) > float(current)].astype(float)
    return dee1_voltage,dee2_voltage

def get_rf_parameters_power(excel_data_df,current):
    forwarded_power = excel_data_df.RF_fwd_W[excel_data_df['Target_I'].astype(float) > float(current)].astype(float)
    reflected_power = excel_data_df.RF_refl_W[excel_data_df['Target_I'].astype(float) > float(current)].astype(float)
    phase_load = excel_data_df.Phase_load[excel_data_df['Target_I'].astype(float) > float(current)].astype(float)
    return forwarded_power,reflected_power,phase_load

def get_rf_parameters_flaps(excel_data_df,current):
    Flap1_pos = excel_data_df.Flap1_pos[excel_data_df['Target_I'].astype(float) > float(current)].astype(float)
    Flap2_pos = excel_data_df.Flap2_pos[excel_data_df['Target_I'].astype(float) > float(current)].astype(float)
    return Flap1_pos,Flap2_pos

def get_magnet_parameters(excel_data_df,current):
    magnet_current = excel_data_df.Magnet_I[excel_data_df['Target_I'].astype(float) > float(current)].astype(float)
    return magnet_current

def get_target_parameters(excel_data_df,current):
    target_current = excel_data_df.Target_I[excel_data_df['Target_I'].astype(float) > float(current)].astype(float)
    return target_current

def get_extraction_parameters(excel_data_df,current):
    extraction_current = excel_data_df.Foil_I[excel_data_df['Target_I'].astype(float) > float(current)].astype(float)
    return extraction_current

def get_extraction_parameters_position(excel_data_df,current):
    carousel_position = excel_data_df.Extr_pos[excel_data_df['Target_I'].astype(float) > float(current)].astype(float)
    balance_position = excel_data_df.Balance[excel_data_df['Target_I'].astype(float) > float(current)].astype(float)
    return carousel_position,balance_position

def get_vacuum_parameters(excel_data_df,current):
    vacuum_level = excel_data_df.Vacuum_P[excel_data_df['Target_I'].astype(float) > float(current)].astype(float)
    return vacuum_level


def get_plots_magnet_all(magnet_current,file_names,output_path):
    fig,ax = plt.subplots()
    print ("archivos")
    print (len(magnet_current))
    for i in range(len(magnet_current)):
         ax.plot(magnet_current[i],label=file_names[i])
    ax.legend(loc='upper left',ncol=5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Magnet current [A]")
    magnetfile = os.path.join(output_path, "magnet_current_evolution_all.png")
    fig.savefig(magnetfile)


def get_plots_gass_all(gas_flow,file_names,output_path):
    fig,ax = plt.subplots()
    print ("archivos")
    print (len(gas_flow))
    for i in range(len(gas_flow)):
         ax.plot(gas_flow[i],label=file_names[i])
    ax.legend(loc='upper left',ncol=5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Gas flow [sccm]")
    gasfile = os.path.join(output_path, "gas_flow_evolution.png")
    fig.savefig(gasfile)

def get_plots_vacuum_all(vacuum_level,file_names,output_path):
    fig,ax = plt.subplots()
    print ("HEREE")
    print (file_names)
    for i in range(len(vacuum_level)):
         time = np.arange(0,len(vacuum_level[i])*3,3)
         ax.plot(time,vacuum_level[i]*1e5,label=file_names[i])
    ax.set_ylabel(r'PRESSURE [$10^{-5}$mbar]')
    ax.legend(loc='lower right',ncol=5)
    ax.set_xlabel("Time [s]")
    vacuumfile = os.path.join(output_path, "vacuum_evolution_all.png")
    fig.savefig(vacuumfile)


def get_plots_collimator(collimator_r_average,collimator_l_average):
    fig,ax = plt.subplots()
    print ("Plot")
    print (collimator_r_average)
    ax.plot(collimator_l_average)
    ax.plot(collimator_r_average)
    ax.set_xlabel("File")
    ax.set_ylabel("Relative current [%]")
    fig.savefig("test_collimator_current.png")

#def get_plots_current(average_current,minumum_current,maximum_current,std_current):
#    fig,ax = plt.subplots()
#    print ("Plot")
#    print (collimator_r_average)
#    ax.plot(collimator_l_average)
#    ax.plot(collimator_r_average)
#    ax.set_xlabel("File")
#    ax.set_ylabel("Relative current [%]")
#    fig.savefig("test_collimator_current.png")

def get_statistic_values(value):
    average_value = (np.mean(value))
    std_value = (np.std(value))
    try:
       max_value = (np.max(value))
       min_value = (np.min(value))
    except:
       max_value = 0
       min_value = 0
    return average_value,std_value,max_value,min_value

def get_plots_extraction(df_all,output_path):
    df_all.set_index("STATISTIC_PARAMETERS",inplace=True)
    df_max = (df_all.loc["MAX"]).to_numpy()
    df_min = (df_all.loc["MIN"]).to_numpy()
    df_std = (df_all.loc["STD"]).to_numpy()
    df_ave = (df_all.loc["AVE"]).to_numpy()
    file = ((np.transpose(np.array(df_max))[0]))
    relative_r = ((np.transpose(np.array(df_max))[2])).astype(np.float)
    min_relative_r = ((np.transpose(np.array(df_min))[2])).astype(np.float)
    ave_relative_r = ((np.transpose(np.array(df_ave))[2])).astype(np.float)
    std_relative_r = ((np.transpose(np.array(df_std))[2])).astype(np.float)
    relative_l = ((np.transpose(np.array(df_max))[3])).astype(np.float)
    min_relative_l = ((np.transpose(np.array(df_min))[3])).astype(np.float)
    ave_relative_l = ((np.transpose(np.array(df_ave))[3])).astype(np.float)
    std_relative_l = ((np.transpose(np.array(df_std))[3])).astype(np.float)
    extraction = ((np.transpose(np.array(df_max))[4])).astype(np.float)
    min_extraction = ((np.transpose(np.array(df_min))[4])).astype(np.float)
    ave_extraction = ((np.transpose(np.array(df_ave))[4])).astype(np.float)
    std_extraction = ((np.transpose(np.array(df_std))[4])).astype(np.float)
    #plt.figure()
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('FILE')
    ax1.set_ylabel('COLLIMATOR RELATIVE CURRENT [%]')
    ax1.plot(file,relative_r, color=COLORS[1],linestyle="--",label="Maximum (Coll r)")
    ax1.plot(file,min_relative_r, color=COLORS[7],linestyle="--",label="Minimum (Coll r)")
    ax1.plot(file,ave_relative_r, color=COLORS[5],linestyle="--",label="Average (Coll r)")
    ax1.plot(file,relative_l, color=COLORS[1],label="Maximum (Coll l)")
    ax1.plot(file,min_relative_l, color=COLORS[7],label="Minimum (Coll l)")
    ax1.plot(file,ave_relative_l, color=COLORS[5],label="Average(Coll l)")
    ax1.fill_between(np.array(file),np.array(ave_relative_r-std_relative_r), np.array(ave_relative_r+std_relative_r),alpha=0.2,color=COLORS[4])
    ax1.fill_between(np.array(file),np.array(ave_relative_l-std_relative_l), np.array(ave_relative_l+std_relative_l),alpha=0.2,color=COLORS[4])
    ax1.tick_params(axis='y')
    fig.tight_layout() 
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend(loc='upper left',ncol=4)   
    extractionfile = os.path.join(output_path, "extraction_evolution.png")
    fig.savefig(extractionfile)  
    fig, ax2 = plt.subplots()  # instantiate a second axes that shares the same x-axis
    ax2.plot(file,extraction, color=COLORS[1],label="Maximum")
    ax2.plot(file,min_extraction, color=COLORS[7],label="Minimum")
    ax2.plot(file,ave_extraction, color=COLORS[5],label="Average")
    ax2.set_ylabel('EFFICIENCY [%]') 
    ax2.set_xlabel('FILE')   # we already handled the x-label with ax1
    ax2.fill_between(np.array(file),np.array(ave_extraction-std_extraction), np.array(ave_extraction+std_extraction),alpha=0.2,color=COLORS[4])    
    #ax2.set_ylim([4.5,5.5])
    #ax2.tick_params(axis='y')
    efficiencyfile = os.path.join(output_path, "efficiency_evolution.png")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped    
    fig.savefig(efficiencyfile)  


def get_plots_extraction_absolute(df_all,output_path):
    df_all.set_index("STATISTIC_PARAMETERS",inplace=True)
    df_max = (df_all.loc["MAX"]).to_numpy()
    df_min = (df_all.loc["MIN"]).to_numpy()
    df_std = (df_all.loc["STD"]).to_numpy()
    df_ave = (df_all.loc["AVE"]).to_numpy()
    file = ((np.transpose(np.array(df_max))[0]))
    current_r = ((np.transpose(np.array(df_max))[2])).astype(np.float)
    min_current_r = ((np.transpose(np.array(df_min))[2])).astype(np.float)
    ave_current_r = ((np.transpose(np.array(df_ave))[2])).astype(np.float)
    std_current_r = ((np.transpose(np.array(df_std))[2])).astype(np.float)
    current_l = ((np.transpose(np.array(df_max))[3])).astype(np.float)
    min_current_l = ((np.transpose(np.array(df_min))[3])).astype(np.float)
    ave_current_l = ((np.transpose(np.array(df_ave))[3])).astype(np.float)
    std_current_l = ((np.transpose(np.array(df_std))[3])).astype(np.float)
    target = ((np.transpose(np.array(df_max))[4])).astype(np.float)
    min_target = ((np.transpose(np.array(df_min))[4])).astype(np.float)
    ave_target = ((np.transpose(np.array(df_ave))[4])).astype(np.float)
    std_target = ((np.transpose(np.array(df_std))[4])).astype(np.float)
    foil = ((np.transpose(np.array(df_max))[5])).astype(np.float)
    min_foil = ((np.transpose(np.array(df_min))[5])).astype(np.float)
    ave_foil = ((np.transpose(np.array(df_ave))[5])).astype(np.float)
    std_foil = ((np.transpose(np.array(df_std))[5])).astype(np.float)
    #plt.figure()
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('FILE')
    ax1.set_ylabel('COLLIMATOR CURRENT [mA]')
    ax1.plot(file,current_r, color=COLORS[1],linestyle="--",label="Maximum (Coll r)")
    ax1.plot(file,min_current_r, color=COLORS[7],linestyle="--",label="Minimum (Coll r)")
    ax1.plot(file,ave_current_r, color=COLORS[5],linestyle="--",label="Average (Coll r)")
    ax1.plot(file,current_l, color=COLORS[1],label="Maximum (Coll l)")
    ax1.plot(file,min_current_l, color=COLORS[7],label="Minimum (Coll l)")
    ax1.plot(file,ave_current_l, color=COLORS[5],label="Average(Coll l)")
    ax1.fill_between(file,np.array(ave_current_r-std_current_r), np.array(ave_current_r+std_current_r),alpha=0.2,color=COLORS[4])
    ax1.fill_between(file,np.array(ave_current_l-std_current_l), np.array(ave_current_l+std_current_l),alpha=0.2,color=COLORS[4])
    ax1.tick_params(axis='y')
    fig.tight_layout() 
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend(loc='upper left',ncol=4)    
    extractionfile = os.path.join(output_path, "absolute_extraction_evolution.png")
    fig.savefig(extractionfile)  
    fig, ax2 = plt.subplots()  # instantiate a second axes that shares the same x-axis
    ax2.set_xlabel('FILE')
    ax2.set_ylabel('CURRENT [mA]')
    ax2.plot(file,target, color=COLORS[1],linestyle="--",label="Maximum (Target)")
    ax2.plot(file,min_target, color=COLORS[7],linestyle="--",label="Minimum (Target)")
    ax2.plot(file,ave_target, color=COLORS[5],linestyle="--",label="Average (Target)")
    ax2.plot(file,foil, color=COLORS[1],label="Maximum (Foil)")
    ax2.plot(file,min_foil, color=COLORS[7],label="Minimum (Foil)")
    ax2.plot(file,ave_foil, color=COLORS[5],label="Average(Foil)")
    ax2.fill_between(np.array(file),np.array(ave_target-std_target), np.array(ave_target+std_target),alpha=0.2,color=COLORS[4])
    ax2.fill_between(np.array(file),np.array(ave_foil-std_foil), np.array(ave_foil+std_foil),alpha=0.2,color=COLORS[4])
    ax2.legend(loc='upper left',ncol=4)
    fig.tight_layout() 
    fig.tight_layout()  # otherwise the right y-label is slightly clipped 
    currenttargetfile = os.path.join(output_path, "current_target_evolution.png")    
    fig.savefig(currenttargetfile)  




def get_plots_vacuum(df_all,output_path):
    df_all.set_index("STATISTIC_PARAMETERS",inplace=True)
    df_max = (df_all.loc["MAX"]).to_numpy()
    df_min = (df_all.loc["MIN"]).to_numpy()
    df_std = (df_all.loc["STD"]).to_numpy()
    df_ave = (df_all.loc["AVE"]).to_numpy()
    file = ((np.transpose(np.array(df_max))[0]))
    vacuum = ((np.transpose(np.array(df_max))[2])).astype(np.float)
    min_vacuum = ((np.transpose(np.array(df_min))[2])).astype(np.float)
    ave_vacuum = ((np.transpose(np.array(df_ave))[2])).astype(np.float)
    std_vacuum = ((np.transpose(np.array(df_std))[2])).astype(np.float)
    #plt.figure()
    fig, ax1 = plt.subplots()
    ax1.ticklabel_format(axis="y",style="sci")
    ax1.set_xlabel('FILE')
    ax1.set_ylabel(r'PRESSURE [$10^{-5}$mbar] ')
    ax1.plot(file,vacuum*1e5, color=COLORS[1],label="Maximum")
    ax1.plot(file,min_vacuum*1e5, color=COLORS[7],label="Minimum")
    ax1.plot(file,ave_vacuum*1e5, color=COLORS[4],label="Average")
    ax1.fill_between(np.array(file),np.array(ave_vacuum-std_vacuum)*1e5, np.array(ave_vacuum+std_vacuum)*1e5,alpha=0.2)
    #ax1.tick_params(axis='y')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped  
    vacuumfile = os.path.join(output_path, "vacuum_evolution.png")
    fig.savefig(vacuumfile)  


def get_plots_magnet(df_all,output_path):
    df_all.set_index("STATISTIC_PARAMETERS",inplace=True)
    df_max = (df_all.loc["MAX"]).to_numpy()
    print(df_max)
    df_min = (df_all.loc["MIN"]).to_numpy()
    df_std = (df_all.loc["STD"]).to_numpy()
    df_ave = (df_all.loc["AVE"]).to_numpy()
    file = ((np.transpose(np.array(df_max))[0]))
    current = ((np.transpose(np.array(df_max))[2])).astype(np.float)
    min_current = ((np.transpose(np.array(df_min))[2])).astype(np.float)
    ave_current = ((np.transpose(np.array(df_ave))[2])).astype(np.float)
    std_current = ((np.transpose(np.array(df_std))[2])).astype(np.float)
    max_current_sorted = [max_current for _,max_current in sorted(zip(file,max_current))]
    min_current_sorted = [min_current for _,min_current in sorted(zip(file,min_current))]
    std_current_sorted = [std_current for _,std_current in sorted(zip(file,std_current))]
    #plt.figure()
    fig, ax1 = plt.subplots()
    ax1.ticklabel_format(axis="y",style="sci")
    ax1.set_xlabel('FILE')
    ax1.set_ylabel(r'CURRENT [A] ')
    ax1.plot(file,max_current_sorted, color=COLORS[1],label="Maximum")
    ax1.plot(file,min_current_sorted, color=COLORS[7],label="Minimum")
    ax1.plot(file,ave_current_sorted, color=COLORS[4],label="Average")
    ax1.fill_between(np.array(file),np.array(ave_current_sorted-std_current_sorted), np.array(ave_current_sorted+std_current_sorted),alpha=0.2)
    #ax1.tick_params(axis='y')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    currentfile = os.path.join(output_path, "magnet_current_evolution.png")
    fig.savefig(currentfile)  

def get_plots_source(df_all,output_path):
    df_all.set_index("STATISTIC_PARAMETERS",inplace=True)
    df_max = (df_all.loc["MAX"]).to_numpy()
    df_min = (df_all.loc["MIN"]).to_numpy()
    df_std = (df_all.loc["STD"]).to_numpy()
    df_ave = (df_all.loc["AVE"]).to_numpy()
    file = ((np.transpose(np.array(df_max))[0]))
    print ("INFORMATION CURRENT")
    print (file)
    file_f = np.sort(((np.transpose(np.array(df_max))[0])))
    file_f2 = np.sort(file)
    current = ((np.transpose(np.array(df_max))[2])).astype(np.float)
    min_current = ((np.transpose(np.array(df_min))[2])).astype(np.float)
    ave_current = ((np.transpose(np.array(df_ave))[2])).astype(np.float)
    std_current = ((np.transpose(np.array(df_std))[2])).astype(np.float)
    gas_flow = ((np.transpose(np.array(df_max))[4])).astype(np.float)
    #plt.figure()
    fig = figure_pz()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xlabel('FILE')
    ax1.set_ylabel('CURRENT [mA]')
    ax1.plot(file_f2.astype(np.float),current, color=COLORS[1],label="Maximum")
    ax1.fill_between(file_f2.astype(np.float),np.array(ave_current-std_current), np.array(ave_current+std_current),alpha=0.2)
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('GAS FLOW [sccm]', color=COLORS[5])  # we already handled the x-label with ax1
    ax2.plot(file_f2.astype(np.float),gas_flow, color=COLORS[5],label="Gas flow")
    ax1.legend(loc='upper left',ncol=4)
    ax2.tick_params(axis='y', labelcolor=COLORS[5])
    fig.tight_layout()  # otherwise the right y-label is slightly clipped 
    sourcefile = os.path.join(output_path, "source_current_evolution.png")
    plt.show()
    fig.savefig(sourcefile)  

def get_box_plots(all_current,all_gas_flow,file_number,output_path):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    # rectangular box plot
    bplot1 = axes.boxplot(all_current,
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=file_number)  # will be used to label x-ticks
    axes.set_title('Rectangular box plot')
    # notch shape box plot
    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen','pink', 'lightblue', 'lightgreen','pink', 'lightblue', 'lightgreen','pink', 'lightblue', 'lightgreen','pink', 'lightblue', 'lightgreen','pink', 'lightblue', 'lightgreen','pink', 'lightblue', 'lightgreen']
    for patch, color in zip(bplot1['boxes'], colors):
            patch.set_facecolor(color)
    
    # adding horizontal grid lines
    axes.yaxis.grid(True)
    axes.set_xlabel('Three separate samples')
    axes.set_ylabel('Observed values')
    sourcefile = os.path.join(output_path, "source_current_evolution_all.png")
    fig.savefig(sourcefile)  

def get_plots_rf(df_all,output_path,file_number):
    df_all.set_index("STATISTIC_PARAMETERS",inplace=True)
    df_max = (df_all.loc["MAX"]).to_numpy()
    df_min = (df_all.loc["MIN"]).to_numpy()
    df_std = (df_all.loc["STD"]).to_numpy()
    df_ave = (df_all.loc["AVE"]).to_numpy()
    file = ((np.transpose(np.array(df_max))[0]))
    file_f = np.sort(((np.transpose(np.array(df_max))[0])))
    file_f2 = np.sort(file_number)
    max_forwarded_power = ((np.transpose(np.array(df_max))[4])).astype(np.float)
    min_forwarded_power = ((np.transpose(np.array(df_min))[4])).astype(np.float)
    std_forwarded_power = ((np.transpose(np.array(df_std))[4])).astype(np.float)
    ave_forwarded_power = ((np.transpose(np.array(df_ave))[4])).astype(np.float)
    max_reflected_power = ((np.transpose(np.array(df_max))[5])).astype(np.float)
    min_reflected_power = ((np.transpose(np.array(df_min))[5])).astype(np.float)
    std_reflected_power = ((np.transpose(np.array(df_std))[5])).astype(np.float)
    ave_reflected_power = ((np.transpose(np.array(df_ave))[5])).astype(np.float)
    max_forwarded_power_sorted = [max_forwarded_power for _,max_forwarded_power in sorted(zip(file,max_forwarded_power))]
    min_forwarded_power_sorted = [min_forwarded_power for _,min_forwarded_power in sorted(zip(file,min_forwarded_power))]
    std_forwarded_power_sorted = [std_forwarded_power for _,std_forwarded_power in sorted(zip(file,std_forwarded_power))]
    ave_forwarded_power_sorted = [ave_forwarded_power for _,ave_forwarded_power in sorted(zip(file,ave_forwarded_power))]
    max_reflected_power_sorted = [max_reflected_power for _,max_reflected_power in sorted(zip(file,max_reflected_power))]
    min_reflected_power_sorted = [min_reflected_power for _,min_reflected_power in sorted(zip(file,min_reflected_power))]
    std_reflected_power_sorted = [std_reflected_power for _,std_reflected_power in sorted(zip(file,std_reflected_power))]
    ave_reflected_power_sorted = [ave_reflected_power for _,ave_reflected_power in sorted(zip(file,ave_reflected_power))]
    print ("INFORMATION")
    print (np.array(file_f))
    print (file_f2)
    print (file.astype(np.float))
    print (ave_reflected_power_sorted)
    max_flap1 = ((np.transpose(np.array(df_max))[7])).astype(np.float)
    min_flap1 = ((np.transpose(np.array(df_min))[7])).astype(np.float)
    std_flap1 = ((np.transpose(np.array(df_std))[7])).astype(np.float)
    ave_flap1 = ((np.transpose(np.array(df_ave))[7])).astype(np.float)
    max_flap2 = ((np.transpose(np.array(df_max))[8])).astype(np.float)
    min_flap2 = ((np.transpose(np.array(df_min))[8])).astype(np.float)
    std_flap2 = ((np.transpose(np.array(df_std))[8])).astype(np.float)
    ave_flap2 = ((np.transpose(np.array(df_ave))[8])).astype(np.float)
    ave_phase_load = ((np.transpose(np.array(df_ave))[4])).astype(np.float)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('FILE')
    ax1.set_ylabel('POWER [kW]')
    ax1.plot(np.array(file_f2),max_forwarded_power_sorted, color=COLORS[1],label="Maximum (Forwarded)")
    ax1.plot(np.array(file_f2),min_forwarded_power_sorted, color=COLORS[7],label="Minimum (Forwarded)")
    ax1.plot(np.array(file_f2),ave_forwarded_power_sorted, color=COLORS[4],label="Average (Forwarded)")
    ax1.plot(np.array(file_f2),max_reflected_power_sorted, color=COLORS[1],linestyle="--",label="Maximum (Reflected)")
    ax1.plot(np.array(file_f2),min_reflected_power_sorted, color=COLORS[7],linestyle="--",label="Minimum (Reflected)")
    ax1.plot(np.array(file_f2),ave_reflected_power_sorted, color=COLORS[4],linestyle="--",label="Average (Reflected)")
    ax1.fill_between(np.array(file_f2),np.array(ave_forwarded_power-std_forwarded_power), np.array(ave_forwarded_power+std_forwarded_power),alpha=0.2,color=COLORS[4])
    ax1.fill_between(np.array(file_f2),np.array(ave_reflected_power-std_reflected_power), np.array(ave_reflected_power+std_reflected_power),alpha=0.2,color=COLORS[4])
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left',ncol=3)
    ax1.set_ylim([-0.5,16])
    powerfile = os.path.join(output_path, "power_evolution.png")
    fig.savefig(powerfile)      
    fig, ax2 = plt.subplots()
    ax2.set_xlabel('FILE')
    ax2.set_ylabel('FLAP POSITION [%]')
    ax2.plot(file.astype(np.float),max_flap1, color=COLORS[1],label="Maximum (Flap 1)")
    ax2.plot(file.astype(np.float),min_flap1, color=COLORS[7],label="Minimum (Flap 1)")
    ax2.plot(file.astype(np.float),ave_flap1, color=COLORS[4],label="Average (Flap 1)")
    ax2.plot(file.astype(np.float),max_flap2, color=COLORS[1],linestyle="--",label="Maximum (Flap 2)")
    ax2.plot(file.astype(np.float),min_flap2, color=COLORS[7],linestyle="--",label="Minimum (Flap 2)")
    ax2.plot(file.astype(np.float),ave_flap2, color=COLORS[4],linestyle="--",label="Average (Flap 2)")
    #ax2.set_ylim([35,40])
    ax2.fill_between(np.array(file.astype(np.float)),np.array(ave_flap1-std_flap1), np.array(ave_flap1+std_flap1),alpha=0.2)
    ax2.fill_between(np.array(file.astype(np.float)),np.array(ave_flap2-std_flap2), np.array(ave_flap2+std_flap2),alpha=0.2)
    ax2.tick_params(axis='y')
    ax2.legend(loc='upper left',ncol=4)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('PHASE LOAD')  # we already handled the x-label with ax1
    ax2.plot((np.transpose(np.array(df_max))[0]),ave_phase_load,label="Phase load")
    ax2.legend(loc='lower right',ncol=4)
    #ax2.set_ylim([4.5,5.5])
    ax2.tick_params(axis='y')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped  
    flapfile = os.path.join(output_path, "flap_evolution.png")
    fig.savefig(flapfile)  


def histogram(input_path,output_path,):
    num_bins = 20
    # the histogram of the data
    n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='blue', alpha=0.5)
    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--')
    plt.xlabel('Ion source current [mA]')
    plt.ylabel('Counts')
    plt.title(r'Histogram of IS current')   
    # Tweak spacing to prevent clipping of ylabel
    sourcefile_histogram = os.path.join(output_path, "histogram_source_current_evolution.png")
    plt.subplots_adjust(left=0.15)
    plt.show()

def main(input_path,output_path,target_current):
    current = target_current
    columns = ["STATISTIC_PARAMETERS","FILE","TARGET","CURRENT","VOLTAGE","HFLOW"] 
    columns_rf =  ["STATISTIC_PARAMETERS","FILE","TARGET","DEE1_VOLTAGE","DEE2_VOLTAGE","FORWARD_POWER","REFLECTED_POWER","PHASE_LOAD","FLAP1","FLAP2"] 
    columns_extraction_values_absolute = ["STATISTIC_PARAMETERS","FILE","TARGET","COLLIMATOR_CURRENT_L","COLLIMATOR_CURRENT_R","TARGET_CURRENT","FOIL_CURRENT","CARROUSEL_POSITION","BALANCE_POSITION"]
    columns_extraction_values_relative = ["STATISTIC_PARAMETERS","FILE","TARGET","RELATIVE_COLLIMATOR_CURRENT","RELATIVE_COLLIMATOR_CURRENT_L","EXTRACTION_LOSSES"]
    columns_vacuum = ["STATISTIC_PARAMETERS","FILE","TARGET","PRESSURE"]
    columns_magnet = ["STATISTIC_PARAMETERS","FILE","TARGET","CURRENT"]
    df_rf_all = pd.DataFrame(columns=columns_rf)
    df_extraction_absolute_all = pd.DataFrame(columns=columns_extraction_values_absolute)
    df_extraction_relative_all = pd.DataFrame(columns=columns_extraction_values_relative)
    df_source_all = pd.DataFrame(columns=columns)
    df_vacuum_all = pd.DataFrame(columns=columns_vacuum)
    df_magnet_all = pd.DataFrame(columns=columns_magnet)
    all_current = []
    all_vacuum = []
    file_number = []
    all_magnet = []
    all_gas_flow = []
    for file in os.listdir(input_path):
        file_path = os.path.join(input_path, file)
        print (file)
        file_number.append(float(file[:-4]))
        real_values,target_number = get_data_tuple(file_path)
        excel_data_df = get_data(real_values)
        time = get_time(excel_data_df,current)
        # source parameters 
        source_voltage,source_current,gas_flow = get_source_parameters(excel_data_df,current)
        all_current.append(source_current)
        all_gas_flow.append(gas_flow)
        # rf parameters
        dee1_voltage,dee2_voltage = get_rf_parameters(excel_data_df,current)
        forwarded_power,reflected_power,phase_load = get_rf_parameters_power(excel_data_df,current)
        Flap1_pos,Flap2_pos = get_rf_parameters_flaps(excel_data_df,current)
        # magnet parameters
        magnet_current = get_magnet_parameters(excel_data_df,current)
        all_magnet.append(magnet_current)
        # extraction and target parameter 
        extraction_current = get_extraction_parameters(excel_data_df,current)
        collimator_r,collimator_l = get_collimator_parameters(excel_data_df,current)
        carousel_position,balance_position = get_extraction_parameters_position(excel_data_df,current)
        target_current = get_target_parameters(excel_data_df,current)
        # vacuum parameters
        vacuum_level = get_vacuum_parameters(excel_data_df,current)
        all_vacuum.append(vacuum_level)
        # summary voltage 
        ave_dee1_voltage,std_dee1_voltage,max_dee1_voltage,min_dee1_voltage = get_statistic_values(dee1_voltage)
        ave_dee2_voltage,std_dee2_voltage,max_dee2_voltage,min_dee2_voltage = get_statistic_values(dee2_voltage)
        ave_forwarded_power,std_forwarded_power,max_forwarded_power,min_forwarded_power = get_statistic_values(forwarded_power)
        ave_reflected_power,std_reflected_power,max_reflected_power,min_reflected_power = get_statistic_values(reflected_power)
        ave_phase_load,std_phase_load,max_phase_load,min_phase_load = get_statistic_values(phase_load)
        ave_Flap1_pos,std_Flap1_pos,max_Flap1_pos,min_Flap1_pos = get_statistic_values(Flap1_pos)
        ave_Flap2_pos,std_Flap2_pos,max_Flap2_pos,min_Flap2_pos = get_statistic_values(Flap2_pos)
        #magnet
        ave_magnet_current,std_magnet_current,max_magnet_current,min_magnet_current = get_statistic_values(magnet_current)
        #extraction
        ave_extraction_current,std_extraction_current,max_extraction_current,min_extraction_current = get_statistic_values(extraction_current)
        ave_collimator_r,std_collimator_r, max_collimator_r, min_collimator_r = get_statistic_values(collimator_r)
        ave_collimator_l,std_collimator_l, max_collimator_l, min_collimator_l = get_statistic_values(collimator_l)
        ave_carousel_position,std_carousel_position, max_carousel_position, min_carousel_position = get_statistic_values(carousel_position)
        ave_balance_position,std_balance_position, max_balance_position, min_balance_position = get_statistic_values(balance_position)
        ave_target_current,std_target_current, max_target_current, min_target_current = get_statistic_values(target_current)
        #verifying current at the collimators 
        relative_current_collimator_l = (((np.array(collimator_l)/np.array(target_current)*100)))
        relative_current_collimator_r = (((np.array(collimator_r)/np.array(target_current)*100)))
        current_losses = ((collimator_l + target_current + collimator_r)/extraction_current)*100
        ave_relative_current_collimator_l,std_relative_current_collimator_l, max_relative_current_collimator_l, min_relative_current_collimator_l = get_statistic_values(relative_current_collimator_l)
        ave_relative_current_collimator_r,std_relative_current_collimator_r, max_relative_current_collimator_r, min_relative_current_collimator_r = get_statistic_values(relative_current_collimator_r)
        ave_current_losses,std_current_losses, max_current_losses, min_current_losses = get_statistic_values(current_losses)
        #vacumm
        ave_vacuum,std_vacuum,max_vacuum,min_vacuum = get_statistic_values(vacuum_level)
        #source 
        ave_source_voltage,std_source_voltage,max_source_voltage,min_source_voltage = get_statistic_values(source_voltage)
        ave_source_current,std_source_current,max_source_current,min_source_current = get_statistic_values(source_current)
        ave_gas_flow,std_gas_flow,max_gas_flow,min_gas_flow = get_statistic_values(gas_flow)
        #vacuum values 
        vacuum_values = [[str("MAX"),"MIN","AVE","STD"],[float(file[:-4]),float(file[:-4]),float(file[:-4]),float(file[:-4])],
        [target_number,target_number,target_number,target_number],[max_vacuum,min_vacuum,ave_vacuum,std_vacuum]]
        #magnet values 
        magnet_values = [[str("MAX"),"MIN","AVE","STD"],[float(file[:-4]),float(file[:-4]),float(file[:-4]),float(file[:-4])],
        [target_number,target_number,target_number,target_number],[max_magnet_current,min_magnet_current,ave_magnet_current,std_magnet_current]]
        #print RF values
        rf_values = [[str("MAX"),"MIN","AVE","STD"],[float(file[:-4]),float(file[:-4]),float(file[:-4]),float(file[:-4])],
        [target_number,target_number,target_number,target_number],
        [max_dee1_voltage,min_dee1_voltage,ave_dee1_voltage,std_dee1_voltage],
        [max_dee2_voltage,min_dee2_voltage,ave_dee2_voltage,std_dee2_voltage],
        [max_forwarded_power,min_forwarded_power,ave_forwarded_power,std_forwarded_power],
        [max_reflected_power,min_reflected_power,ave_reflected_power,std_reflected_power],
        [max_phase_load,min_phase_load,ave_phase_load,std_phase_load],
        [max_Flap1_pos,min_Flap1_pos,ave_Flap1_pos,std_Flap1_pos],
        [max_Flap2_pos,min_Flap2_pos,ave_Flap2_pos,std_Flap2_pos]]
        # print extraction absolute
        extraction_values_absolute = [[str("MAX"),"MIN","AVE","STD"],[float(file[:-4]),float(file[:-4]),float(file[:-4]),float(file[:-4])],
        [target_number,target_number,target_number,target_number],
        [max_collimator_r,min_collimator_r,ave_collimator_r,std_collimator_r],
        [max_collimator_l,min_collimator_l,ave_collimator_l,std_collimator_l],
        [max_target_current,min_target_current,ave_target_current,std_target_current],
        [max_extraction_current,min_extraction_current,ave_extraction_current,std_extraction_current],
        [max_carousel_position,min_carousel_position,ave_carousel_position,std_carousel_position],
        [max_balance_position,min_balance_position,ave_balance_position,std_balance_position]]
        # print extraction relative
        extraction_values_relative = [[str("MAX"),"MIN","AVE","STD"],[float(file[:-4]),float(file[:-4]),float(file[:-4]),float(file[:-4])],
        [target_number,target_number,target_number,target_number],
        [max_relative_current_collimator_l, min_relative_current_collimator_l,ave_relative_current_collimator_l,std_relative_current_collimator_l],
        [max_relative_current_collimator_r, min_relative_current_collimator_r,ave_relative_current_collimator_r,std_relative_current_collimator_r],
        [max_current_losses, min_current_losses,ave_current_losses,std_current_losses]]
        # print source values
        source_values = [[str("MAX"),"MIN","AVE","STD"],[float(file[:-4]),float(file[:-4]),float(file[:-4]),float(file[:-4])],
        [target_number,target_number,target_number,target_number],
        [float(max_source_current), float(min_source_current),float(ave_source_current),float(std_source_current)],
        [float(max_source_voltage), float(min_source_voltage),float(ave_source_voltage),float(std_source_voltage)],
        [float(max_gas_flow), float(min_gas_flow),float(ave_gas_flow),float(std_gas_flow)]]
        # print dataframe 
        df_source = pd.DataFrame(np.transpose(np.array(source_values)),columns=columns)
        df_rf = pd.DataFrame(np.transpose(np.array(rf_values)),columns=columns_rf)
        df_extraction_relative = pd.DataFrame(np.transpose(np.array(extraction_values_relative)),columns=columns_extraction_values_relative)
        df_extraction_absolute = pd.DataFrame(np.transpose(np.array(extraction_values_absolute)),columns=columns_extraction_values_absolute)
        df_vacuum = pd.DataFrame(np.transpose(np.array(vacuum_values)),columns=columns_vacuum)
        df_magnet = pd.DataFrame(np.transpose(np.array(magnet_values)),columns=columns_magnet)
        df_rf_all = df_rf_all.append(df_rf, ignore_index=True)
        df_extraction_absolute_all = df_extraction_absolute_all.append(df_extraction_absolute, ignore_index=True)
        df_extraction_relative_all = df_extraction_relative_all.append(df_extraction_relative, ignore_index=True)
        df_source_all = df_source_all.append(df_source,ignore_index=True)
        df_vacuum_all = df_vacuum_all.append(df_vacuum,ignore_index=True)
        df_magnet_all = df_magnet_all.append(df_magnet,ignore_index=True)
    #get_plots_extraction(df_extraction_relative_all,output_path)
    #get_plots_extraction_absolute(df_extraction_absolute_all,output_path)
    #get_plots_vacuum_all(all_vacuum,file_number,output_path)
    #get_plots_magnet_all(all_magnet,file_number,output_path)
    #get_plots_gass_all(all_gas_flow,file_number,output_path)
    #get_plots_rf(df_rf_all,output_path,file_number)
    #get_plots_source(df_source_all,output_path)
    df_source_all_n = df_source_all.dropna()
    df_vacuum_all_n = df_vacuum_all.dropna()
    df_magnet_all_n = df_magnet_all.dropna()
    df_rf_all_n = df_rf_all.dropna()
    df_extraction_relative_n = df_extraction_relative.dropna()
    df_extraction_absolute_n = df_extraction_absolute.dropna()
    print (df_magnet_all.columns)
    #tfs_pandas.write_tfs("test_file.out",df_source_all)
    list_dfs = [df_vacuum_all,df_magnet_all,df_rf_all,df_source_all_n,df_extraction_relative_all,df_extraction_absolute_all]
    list_names = ["VACUMM","MAGNET","RF","ION_SOURCE","EXTRACTION RELATIVE","EXTRACTION ABSOLUTE"]
    excel_output = os.path.join(output_path,"excel_summary.xlsx")
    with ExcelWriter(excel_output) as writer:
        for n, df in enumerate(list_dfs):
            print (n)
            print (df)
            df.to_excel(writer,list_names[n])
        writer.save()
    print ("HERE")    
    datas = pd.read_excel(excel_output,sheet_name="ION_SOURCE")
    datas.set_index("STATISTIC_PARAMETERS",inplace=True)
    print (datas)
    print ((datas.loc["MAX"]).to_numpy())

if __name__ == "__main__":
    _input_path,_output_path,target_current = _parse_args()
    main(_input_path,_output_path,target_current)
