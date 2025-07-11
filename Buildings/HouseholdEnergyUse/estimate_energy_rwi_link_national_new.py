from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pathlib
import seaborn as sns
from scipy.stats import norm
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
import sys

# Define probability distribution for selection of simulated subgroups of households from survey
def selection_window(x, x0, s, c):
    p = norm.pdf(x, loc=x0, scale=s)
    p[x > x0+c*s] = 0
    p = p/sum(p)
    return p


# Define fitting function to characterise rwi-f_elec-energy relationship - only used if simulate_cell_groups=False
def logistic(x, a, b, x0):
    # Note that "x" is 2D, containing both the rwi values and f_elec (access rate) values 
    return x[1]*a/(1+b*np.exp(-(x[0]-x0)))


np.random.seed(42)


def estimate_energy_rwi_link_national(grid, app_config):
    np.random.seed(42)

    # To produce graphs of the simulated cell groups, set simulate_cell_groups = True
    simulate_cell_groups = app_config.DHS_SIMULATE_CELL_GROUPS # Setting to False will set cell energies by interpolation (only active if recalculate_energies = True)
    # To produce graphs showing the fitting function, set simulate_cell_groups = False and recalculate_energies = True
    recalculate_energies = app_config.DHS_RECALCULATE_ENERGIES # If false will just use any existing value in grid data
    recalculate_energy_perhh = app_config.DHS_RECALCULATE_ENERGY_PERHH
    figures_folder = app_config.FIGURES_DHS_FOLDER
    make_figure = app_config.DHS_MAKE_FIGURE

    if recalculate_energy_perhh:
        from Buildings.HouseholdEnergyUse.estimate_energy_perhh_DHS import compute_energy_perhh_DHS
        compute_energy_perhh_DHS(app_config)  # Run the script to assess energy consumption of households in the DHS dataset

    # Read file containing data from DHS survey of households
    infile_DHS = app_config.DHS_HOUSEHOLD_DATA_CSV
    dataDHS_all = read_csv(infile_DHS)
    # wealth_index = 1e-5 * dataDHS_all["Wealth index factor score for urban/rural (5 decimals)"].to_numpy(float)
    wealth_index = 1e-5 * dataDHS_all["Wealth index factor score combined (5 decimals)"].to_numpy(float)
    if make_figure:
        min_wealth = wealth_index.min()
        max_wealth = wealth_index.max()
        xl = np.array([min_wealth,max_wealth])  # Limits for x-axis of plots (wealth index)
        yl = np.array([0, dataDHS_all[app_config.DHS_ELEC_KWH_ASSESSED_SURVEY].max()])  # Limits for y-axis on scatter plots (energy use)

    legend_loc = ['lower right','upper left']

    if recalculate_energies:
        N = grid.shape[0]
        energy_demand = np.zeros((2, N))  # Array to save energy demand estimates
        rwi_simulated_group = np.zeros((2, N))

        group_sigma = 1 # in units of rwi
        tail_cutoff = 4  # Number of standard deviations at which selection window is cropped - to allow cells with very low rwi to be matched 
        group_size = 100 # Simulating true group size would be a computational burden and not alter the results
        # Create Nb groups of points with ascending average rwi to get approximate rwi-E mapping
        Nb = 20  # Number of bins in wealth index
        Na = 20 # Number of bins in access rate
        x = np.arange(xl[0],xl[1],0.1) # array to plot fitting function (if needed)

    residence_types = ['urban', 'rural']
    i = 0 # used for figures
    for residence_type in residence_types:

        # Data from DHS dataset
        dataDHS = dataDHS_all[dataDHS_all["Type of place of residence"] == residence_type]
        elec = dataDHS["Electricity"].to_numpy(int)
        # Find households with or without access
        has_access = np.flatnonzero(elec>0)
        no_access = np.flatnonzero(elec==0)
        rwi_DHS = 1e-5 * dataDHS["Wealth index factor score for urban/rural (5 decimals)"].to_numpy(float)
        eu = dataDHS[app_config.DHS_ELEC_KWH_ASSESSED_SURVEY].to_numpy(float)

        # Data from the grid
        col_HH = 'HH_' + residence_type
        col_HH_access = 'HHwithAccess_' + residence_type[:3]
        include = np.flatnonzero(grid[col_HH_access]>0)
        f_elec = grid[col_HH_access][include].to_numpy(float)/grid[col_HH][include].to_numpy(float)
        rwi_grid = grid[app_config.COL_RWI_MEAN][include].to_numpy(float)

        if recalculate_energies:
            # Select centers of selection probability distribution
            # such that the lowest selection window just overlaps the grid mean rwi values
            rwi = np.linspace(np.ceil(rwi_DHS[has_access].min()*10)/10-tail_cutoff,
                            np.ceil(rwi_DHS[no_access].max()*10)/10+group_sigma,
                            Nb)
            # Create factor f going from minimum access rate to maximum
            f = f_elec.min() + np.arange(Na)/(Na-1) * (f_elec.max()-f_elec.min())
            # Prepare array for groups of households
            group = np.zeros((Na, Nb, group_size), dtype=int)
            for k in range(Nb):
                # Create subsample of survey households
                # with peak of selection function = rwi[k]... 
                pn = selection_window(rwi_DHS[no_access],rwi[k],group_sigma,tail_cutoff)
                pa = selection_window(rwi_DHS[has_access],rwi[k],group_sigma,tail_cutoff)
                for j in range(Na):
                    # ...and number of no_access and has_access housholds set to match f[j]
                    group[j,k,:] = np.append(
                        np.random.choice(no_access,int(round(group_size*(1-f[j]),0)),p=pn),
                        np.random.choice(has_access,int(round(group_size*f[j],0)),p=pa)
                    )
            # Calculation average rwi, average energy use for each group
            # At present these groups are just for mapping the parameter space
            rwi_group = np.nanmean(rwi_DHS[group],axis=2)
            eu_group = np.nanmean(eu[group],axis=2)
            f_group = np.sum(elec[group],axis=2)/group_size
            
            # The above code has created a look-up table between mean group rwi (r_group), mean group access rate (f_group), and peak of selection window (rwi)
            if simulate_cell_groups:
                # 1. Create a mask to find rows where rwi_grid are finite
                finite_mask = np.isfinite(rwi_grid)
                # You can add a print statement here for debugging to see if you are losing data
                print(f"Found {np.sum(finite_mask)} finite data points out of {len(rwi_grid)}")
                # 2. Create "clean" versions of the arrays for the griddata input
                rwi_grid_clean = rwi_grid[finite_mask]
                f_elec_clean = f_elec[finite_mask]

                # Now we use this to set the selection function peak that will closely match each cell's values of rwi_grid and f_elec
                rwi_peak_clean = griddata(np.stack((rwi_group.flatten(),f_group.flatten()),axis=1),
                                                    np.array(Na*[rwi]).flatten(),
                                                    np.stack((rwi_grid_clean,f_elec_clean),axis=1),
                                                    method='nearest')
                # points_source = np.stack((rwi_group.flatten(), f_group.flatten()), axis=1)
                # values_source = np.array(Na*[rwi]).flatten()
                #
                # # Now, call griddata ONLY with the clean data
                # rwi_peak_clean = griddata(points_source,
                #                           values_source,
                #                           np.stack((rwi_grid_clean, f_elec_clean), axis=1),
                #                           method='nearest')
                # 3. Create a full-size rwi_peak array and place the clean results back
                # This ensures alignment with the original grid.
                rwi_peak = np.full(rwi_grid.shape, np.nan)
                rwi_peak[finite_mask] = rwi_peak_clean
                # Prepare array to store simulated groups of households for each cell
                group = np.zeros((rwi_peak.size,group_size),dtype=int)
                # Create subsamples of survey households
                for k in range(rwi_peak.size):
                    if np.isfinite(rwi_peak[k]):
                        pn = selection_window(rwi_DHS[no_access],rwi_peak[k],group_sigma,tail_cutoff)
                        pa = selection_window(rwi_DHS[has_access],rwi_peak[k],group_sigma,tail_cutoff)
                        if pn.sum() > 0 and pa.sum() > 0:
                            group[k,:] = np.append(
                                np.random.choice(no_access, int(round(group_size*(1-f_elec[k]), 0)), p=pn),
                                np.random.choice(has_access, int(round(group_size*f_elec[k], 0)), p=pa)
                            )
                        # group[k,:] = np.append(
                        #     np.random.choice(no_access,int(round(group_size*(1-f_elec[k]),0)),p=pn),
                        #     np.random.choice(has_access,int(round(group_size*f_elec[k],0)),p=pa)
                        # )
                # Calculation average rwi, average energy use for each group
                # Now these groups are the simulated groups that match the cell averages
                # rwi_group = np.nanmean(rwi_DHS[group],axis=1)
                # eu_group = np.nanmean(eu[group],axis=1)
                # f_group = np.sum(elec[group],axis=1)/group_size
                # Use np.nanmean to safely ignore empty/invalid groups
                with np.errstate(invalid='ignore', divide='ignore'): # Suppress warnings from mean of empty slice
                    rwi_group = np.nanmean(np.where(group > 0, rwi_DHS[group], np.nan), axis=1)
                    eu_group = np.nanmean(np.where(group > 0, eu[group], np.nan), axis=1)
                    f_group = np.nansum(np.where(group > 0, elec[group], 0), axis=1) / group_size
                # Update global arrays
                energy_demand[i,include] = eu_group.copy()
                rwi_simulated_group[i,include] = rwi_group.copy()
                # Add/update group parameters in grid data
                grid['Simulated group rwi '+residence_type] = rwi_simulated_group[i,:]
            else:
                # To avoid simulating groups for each cell, energy use can be approximated by 
                # using the closest match to rwi_grid and f_elec from the groups created above
                energy_demand[i,include] = griddata(np.stack((rwi_group.flatten(),f_group.flatten()),axis=1),
                                                    eu_group.flatten(),
                                                    np.stack((rwi_grid,f_elec),axis=1),
                                                    method='nearest')
                param = [ 4000,0.5,5]
                try:
                    param,cov = curve_fit(logistic,np.stack((rwi_group.flatten(),f_group.flatten())),eu_group.flatten(),p0 = param)
                except RuntimeError:
                    print('Unable to fit rwi vs eu')
            grid['elec_demand_kWh_'+residence_type] = energy_demand[i,:]
        else:
            eu_group = grid['elec_demand_kWh_'+residence_type][include]
            rwi_group = grid['Simulated group rwi '+residence_type][include]
            f_group = f_elec.copy()
        Etot = (grid['elec_demand_kWh_'+residence_type]*grid[col_HH]).sum()
        print(residence_type+' total = {:,.0f} GWh/year'.format(Etot*1e-6))
        print(residence_type+' average per household = {:,.0f} kWh/year'.format(Etot/grid[col_HH_access].sum()))
        print(residence_type + ' min = {:,.0f} kWh/year'.format(grid['elec_demand_kWh_'+residence_type].min()) +
              ' max = {:,.0f} kWh/year'.format(grid['elec_demand_kWh_'+residence_type].max()))
        
        if make_figure:

            # Set options
            x_axis_cells_results_option = False # option to choose a different x-axis scale
            show_access_rate = False # False will show energy use in the scatter panel

            # Set parameters
            bx = np.arange(xl[0], xl[1], 0.15)  # Bins for histograms on x-axis
            al = 0.3  # Alpha value for points in figures
            letters = ['(a)', '(b)']
            hmax = 25 # Limit for histogram y-axes

            labels = ['DHS survey individual households',
                        'Groups of households for each hexagon cell',
                        'Simulated groups of survey households selected to\nmatch relative wealth index and access rate of each cell']
            legend_fontsize = 8
            palette = sns.color_palette()
            suffix = '' # default suffix of figure png filename

            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 5),
                                            gridspec_kw={'height_ratios': [1, 2]})

            # Top histogram subplot, ax1
            if x_axis_cells_results_option is True:
                # option to choose a different x-axis
                # Parameters for the graphs
                min_rwi = np.floor(grid[app_config.COL_RWI_MEAN][include].min())
                max_rwi = np.ceil(grid[app_config.COL_RWI_MEAN][include].max())
                print(min_rwi, max_rwi)
                xl = np.array([min_rwi, max_rwi])  # Limits for x-axis of plots (wealth index)
                ax1.set_xlim(xl)
            else:
                ax1.set_xlim(xl)
            ax1.set_xticks([])
            ax1.set_ylabel('Percentage\nof households (%)')

            # Plot a weighted density histogram of wi DHS values for survey households.
            w = 1e-6* dataDHS["Household sample weight (6 decimals)"].to_numpy(float)
            ax1.hist(rwi_DHS, bins=bx, weights=w/w.sum()*100, density=False, edgecolor=palette[0],
                        histtype='step',facecolor='None', label=labels[0])
            # Plot a density histogram of rwi values for the grid
            pct_households_in_group = grid[col_HH][include]/grid[col_HH][include].sum()*100
            ax1.hist(rwi_grid,
                        bins=bx, weights=pct_households_in_group,histtype='step',
                        density=False, edgecolor=palette[4], facecolor='None', label=labels[1])
            if simulate_cell_groups:
                ax1.hist(rwi_group,
                        bins=bx, weights=pct_households_in_group,histtype='bar',
                        density=False, edgecolor='None', facecolor=palette[1], label=labels[2],alpha=0.3)
            elif recalculate_energies:
                suffix = 'fit_'
            ax1.legend(fontsize=legend_fontsize)

            # Bottom scatter subplot, ax2
            ax2.set_xlim(xl)
            ax2.set_xlabel('Wealth index')
            if show_access_rate:
                y_value = 'access_rate'
                ax2.set_ylabel('% with electricity access')
                ax2.set_ylim(0,1)
                # Plot a scatter plot of rwi vs. y value for survey households from DHS data
                ax2.scatter(rwi_grid, f_elec, marker='s', alpha=al, c=[palette[4]], edgecolors='None',
                        label=labels[1])
                y_group = f_group
            else:
                y_value = 'energy_use'
                ax2.set_ylabel('Household annual\nelectricity consumption (kWh)')
                ax2.set_ylim(yl)
                ax2.scatter(rwi_DHS, eu, s=w*15, alpha=al, c=[palette[0]], edgecolors='None', label=labels[0])
                ax2.get_yaxis().set_major_formatter(
                    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
                y_group = eu_group
                if not(simulate_cell_groups):
                    # Fit logistic function to simulated groups and plot
                    label_fit = 'Logistic fit at given access rate'
                    label_group = 'Simulated household groups with access rate = '
                    if i == 0:
                        j_fit = [0,-1]
                    else:
                        j_fit = np.arange(4,Na,4)
                    k = 1
                    for j in j_fit:
                        fit = logistic(np.stack((x,x.size*[f[j]])),*param)
                        ax2.plot(x,fit,'-',label=label_fit,alpha=0.7,color=palette[k])
                        label_fit = ''
                        ax2.scatter(rwi_group[j,:], y_group[j,:], marker='d', edgecolors='None',color=palette[k],
                        label=label_group+'{:.2f}'.format(f[j]),alpha=0.7)
                        label_group = label_group.replace('Simulated household groups with ','')
                        k +=1
                    legend_loc[0] = 'upper right'
            # Plot mean rwi and mean y value of the simulated groups
            if simulate_cell_groups:
                ax2.scatter(rwi_group, y_group, marker='d', alpha=al, c=[palette[1]], edgecolors='None',
                        label=labels[2])
            ax2.legend(loc=legend_loc[i],fontsize=legend_fontsize)

            outfile = 'rwi_vs_'+y_value+'_'+ suffix + residence_type +'.png'
            pathlib.Path(figures_folder).mkdir(exist_ok=True)
            fig.suptitle(f'{letters[i]} {residence_type}')
            plt.tight_layout()
            plt.savefig(figures_folder / outfile, dpi=300)
            print('Created ' + outfile)
            plt.show()
            plt.close()

        i = i + 1

            # # Add the assessed energy use to the grid
    return grid


if __name__ == "__main__":
    sys.path.insert(1, '../../')
    import config
    infile = config.RESIDENTIAL_GRID_FILE  # Read file containing the mean relative wealth index ("rwi") of each hexagon on map
    grid = read_csv(infile)

    estimate_energy_rwi_link_national(grid, config)
    grid.to_csv(infile,index=False)

