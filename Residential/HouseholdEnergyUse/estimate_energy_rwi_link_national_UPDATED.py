from pandas import read_csv
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns
from scipy.stats import norm

np.random.seed(42)

data_folder = '../Data/DHSSurvey/'
figures_folder = '../Figures/'
outputs_folder = '../Outputs/'
make_figure = True

elas = 0.35  # choose elasticity value for the country
recalculate_energy_perhh = False
if recalculate_energy_perhh:
    from estimate_energy_perhh_DHS import compute_energy_perhh_DHS
    compute_energy_perhh_DHS(elas=elas)  # Run the script to assess energy consumption of households in the DHS dataset

infile = data_folder + 'household_data.csv'  # Read file containing data from DHS survey of households
dataDHS = read_csv(infile)

wealth_index = 1e-5 * dataDHS["Wealth index factor score for urban/rural (5 decimals)"].to_numpy(float)
weight = 1e-6 * dataDHS['Household sample weight (6 decimals)'].to_numpy(float)
energy_use = dataDHS["Energy Use Elasticity"].to_numpy(float)  # Choose if assessed energy with elas is used or not
# energy_use = dataDHS["Energy Use"].to_numpy(float)
province = dataDHS['Province'].to_numpy(int)

region_type = ['urban', 'rural']

# Parameters for the graphs
min_wealth = np.floor(wealth_index.min())
max_wealth = np.ceil(wealth_index.max())
print(min_wealth, max_wealth)
# xl = np.array([-2, 2])  # Limits for x-axis of plots (wealth index)
xl = np.array([min_wealth, max_wealth])  # Limits for x-axis of plots (wealth index)
bx = np.arange(xl[0], xl[1], 0.15)  # Bins for histograms on x-axis
al = 0.3  # Alpha value for points in figures
x_axis_cells_results_option = False
letters = ['(a)', '(b)']
# Maximum heights for y axes so that both figures are the same; just too much hassle to automate 
yl = np.array([0, np.max(energy_use)])  # Limits for y-axis on scatter plots (energy use)
hmax = 25 # Limit for histogram y-axes

labels = ['DHS individual households',
          'Groups of households inferred from map data',
          'Groups simluated from DHS households\nto match wealth index of map groups']

recalculate_energies = False

hh_cells_results_available = True # Change to False if this is the first run of the analysis
if hh_cells_results_available:
    infile = outputs_folder + 'data.csv'  # Read file containing the mean wealth index ("rwi") of each hexagon on map
    data_hh = read_csv(infile)
    N = data_hh.shape[0]
    # print(N)
    energy_demand = np.zeros((2, N))  # Array to save energy demand estimates
    rwi_simulated_group =  np.zeros((2, N))
    # print(energy_demand)

group_sigma = 1 # in units of rwi
group_size = 100 # Simluating true group size would be huge computational burden not alter the results

for i in range(2):

    this_region_type = dataDHS["Type of place of residence"] == i + 1
    column_name = 'HH_' + region_type[i].lower()
    in_region_type = np.flatnonzero(this_region_type)

    rwi_DHS = wealth_index[in_region_type]
    Nh = rwi_DHS.size
    eu = energy_use[in_region_type]
    w = weight[in_region_type]
    
    if recalculate_energies or not(hh_cells_results_available):

        # Create Nb groups of points with ascending average rwi
        Nb = 20  # Number of artifical groups used to map group wealth index to group energy use
        group = np.zeros((Nb, group_size), dtype=int)
        rwi = np.linspace(min_wealth,max_wealth,Nb)
        for k in range(Nb):
            p = norm.pdf(rwi_DHS,rwi[k],group_sigma)
            group[k, :] = np.random.choice(Nh,group_size,p=p/sum(p))
        # Calculation average rwi, average energy use for each group
        rwi_group = np.nanmean(rwi_DHS[group],axis=1)
        eu_group = np.nanmean(eu[group],axis=1)

        if hh_cells_results_available:
            # Create filter to identify map regions (hexagons) of the relevant type and province
            include = np.flatnonzero((data_hh[column_name] > 0))
    
            Nb = include.size
            group = np.zeros((Nb, group_size), dtype=int)
            rwi_peak = np.interp(data_hh['rwi'][include],rwi_group,rwi)
            for j in range(Nb):
                p = norm.pdf(rwi_DHS,rwi_peak[j],group_sigma)#/dn_drwi # generates random sample with mean close to dataset mean
                group[j,:] = np.random.choice(Nh,group_size,p=p/sum(p))
            # Calculation average rwi, average energy use for each group
            rwi_group = np.nanmean(rwi_DHS[group],axis=1)
            eu_group = np.nanmean(eu[group],axis=1)
        
            # Allocate estimated average household energy demand to hexagon subregions
            # by interpolating between running average of survey data
            # energy_demand[i, include] = np.interp(data_hh['rwi'][include], rwi_group, eu_group)
            energy_demand[i,include] = eu_group.copy()
            rwi_simulated_group[i,include] = rwi_group.copy()
            data_hh['Energy demand '+region_type[i].lower()] = energy_demand[i,:]
            data_hh['Simulated group rwi '+region_type[i].lower()] = rwi_simulated_group[i,:]

    elif hh_cells_results_available:
        include = np.flatnonzero((data_hh[column_name] > 0))
        eu_group = data_hh['Energy demand '+region_type[i].lower()][include]
        rwi_group = data_hh['Simulated group rwi '+region_type[i].lower()][include]

    if make_figure:
        palette = sns.color_palette()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 5),
                                       gridspec_kw={'height_ratios': [1, 2]})

        # First subplot
        # Plots a weighted density histogram of rwi DHS values for survey households.
        ax1.hist(rwi_DHS, bins=bx, weights=w/w.sum()*100, density=False, edgecolor=palette[0],
                 histtype='step',facecolor='None', label=labels[0])
        # sns.kdeplot(x=rwi_DHS, weights=w/w.sum()*100, color=palette[0], ax=ax1, label='Survey households')
        ax1.set_xlim(xl)
        ax1.set_xticks([])
        ax1.set_ylabel('Percentage\nof households')

        if hh_cells_results_available is True:
            if x_axis_cells_results_option is True:
                # option to choose a different x-axis
                # Parameters for the graphs
                min_rwi = np.floor(data_hh['rwi'][include].min())
                max_rwi = np.ceil(data_hh['rwi'][include].max())
                print(min_rwi, max_rwi)
                xl = np.array([min_rwi, max_rwi])  # Limits for x-axis of plots (wealth index)
                ax1.set_xlim(xl)
            # Plots a density histogram of rwi values for our dataset
            pct_households_in_group = data_hh[column_name][include]/data_hh[column_name][include].sum()*100
            ax1.hist(data_hh['rwi'][include],
                     bins=bx, weights=pct_households_in_group,histtype='step',
                     density=False, edgecolor=palette[4], facecolor='None', label=labels[1])
            ax1.hist(rwi_group,
                     bins=bx, weights=pct_households_in_group,histtype='bar',
                     density=False, edgecolor='None', facecolor=palette[1], label=labels[2],alpha=0.3)
        ax1.legend()

        # Second subplot
        ax2.set_xlim(xl)
        ax2.set_ylim(yl)
        ax2.set_ylabel('Household annual\nelectricity consumption (kWh)')
        ax2.set_xlabel('Wealth index')
        scaling = 3
        # Plots a scatter plot of rwi vs. energy use for survey households from DHS data
        ax2.scatter(rwi_DHS, eu, s=w*5*scaling, alpha=al, c=[palette[0]], edgecolors='None', label=labels[0])
        ax2.text(xl[0], yl[-1], '\n  ', va='top')
        # Add the simulated groups
        ax2.scatter(rwi_group, eu_group, marker='d', alpha=al, c=[palette[1]], edgecolors='None',
                    label=labels[2])
        ax2.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax2.legend(loc='upper left')

        if hh_cells_results_available is True:
            outfile = f'household_groups_{region_type[i].lower()}_withweight_elas{elas}_withWorldpopData_newvalues.png'
        else:
            outfile = f'household_groups_{region_type[i].lower()}_withweight_elas{elas}_woWorldpopData_newvalues.png'
        pathlib.Path(figures_folder).mkdir(exist_ok=True)
        fig.suptitle(f'{letters[i]} {region_type[i].capitalize()}')
        plt.tight_layout()
        plt.savefig(figures_folder + outfile, dpi=300)
        print('Created ' + outfile)
        # plt.show()
        plt.close()

    # # Add the assessed energy use in the output file
    if recalculate_energies and hh_cells_results_available:
        data_hh.to_csv(infile)
        print('Written energies to',infile)
    

