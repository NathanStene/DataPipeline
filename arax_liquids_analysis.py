import pandas as pd
import scipy as sp
import statistics
from constants import arax_pcr1_column_names, arax_pcr2_column_names
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import os
from os.path import dirname, expanduser

working_dir = os.getcwd()
home = expanduser('~')
#arax_file_path = os.path.join(home, "OneDrive - miDIAGNOSTICS\\Desktop\\Team\\Projects\\NA\\PCR die\\ARAX")
arax_file_path = os.path.join(home, "miDIAGNOSTICS\Michela Cagna - ARAX")
arax_full_wafer_excel_file_name = 'Arax PCR.xlsx'
excel_file_path = os.path.join(arax_file_path, arax_full_wafer_excel_file_name)

df_0 = pd.read_excel(io=excel_file_path, sheet_name=0, header=1)

# Clean dataset
pcr1_df = df_0.iloc[:, 0:23]
pcr2_df = df_0.iloc[:, np.r_[0:9, 23:38]]

pcr1_df.columns = arax_pcr1_column_names
pcr2_df.columns = arax_pcr2_column_names

pcr1_df.insert(14, 'bubble_splitter', -1)
pcr1_df.insert(4, 'die', 2)
pcr2_df.insert(4, 'die', 3)

pcr_df = pd.concat([pcr1_df, pcr2_df])

pcr_df.replace('na', np.nan, inplace=True)

#c = pcr_df.columns
#pcr_df[[c[15], c[20]]] = pcr_df[[c[20], c[15]]]
#col_list = list(pcr_df)
#col_list[15], col_list[20] = col_list[20], col_list[15]
#pcr_df.columns = col_list

inlet = pcr_df['bubble_inlet']
pcr_df.drop(labels=['bubble_inlet'], axis=1, inplace=True)
pcr_df.insert(15, 'bubble_inlet', inlet)

timing_end = pcr_df['timing_end']
pcr_df.drop(labels=['timing_end'], axis=1, inplace=True)
pcr_df.insert(15, 'timing_end', timing_end)

pcr_df.to_pickle(os.path.join(working_dir, 'pickles\\pcr_df.pkl'))

# Data analysis
# Stats
stat_pcr1_na_wafer = pcr_df.loc[(pcr_df['die'] == 2) &
                                (pcr_df['testing_day'] == 'd0') &
                                (pcr_df['liquid'] == 'NA buffer') &
                                (pcr_df['testing_level'] == 'wafer') &
                                (pcr_df['hole'] == 0), 'timing'].dropna()

stat_pcr2_na_wafer = pcr_df.loc[(pcr_df['die'] == 3) &
                                (pcr_df['testing_day'] == 'd0') &
                                (pcr_df['liquid'] == 'NA buffer') &
                                (pcr_df['testing_level'] == 'wafer') &
                                (pcr_df['hole'] == 0), 'timing'].dropna()

stat_pcr1_nacl_wafer = pcr_df.loc[(pcr_df['die'] == 2) &
                                  (pcr_df['testing_day'] == 'd0') &
                                  (pcr_df['liquid'] == 'NaCl 0.9%') &
                                  (pcr_df['testing_level'] == 'wafer') &
                                  (pcr_df['hole'] == 0), 'timing'].dropna()

stat_pcr2_nacl_wafer = pcr_df.loc[(pcr_df['die'] == 3) &
                                  (pcr_df['testing_day'] == 'd0') &
                                  (pcr_df['liquid'] == 'NaCl 0.9%') &
                                  (pcr_df['testing_level'] == 'wafer') &
                                  (pcr_df['hole'] == 0), 'timing'].dropna()

stat_pcr1_na_die = pcr_df.loc[(pcr_df['die'] == 2) &
                              (pcr_df['testing_day'] == 'd0') &
                              (pcr_df['liquid'] == 'NA buffer') &
                              (pcr_df['testing_level'] == 'die') &
                              (pcr_df['hole'] == 0), 'timing'].dropna()

stat_pcr2_na_die = pcr_df.loc[(pcr_df['die'] == 3) &
                              (pcr_df['testing_day'] == 'd0') &
                              (pcr_df['liquid'] == 'NA buffer') &
                              (pcr_df['testing_level'] == 'die') &
                              (pcr_df['hole'] == 0), 'timing'].dropna()

stat_pcr1_nacl_die = pcr_df.loc[(pcr_df['die'] == 2) &
                                (pcr_df['testing_day'] == 'd0') &
                                (pcr_df['liquid'] == 'NaCl 0.9%') &
                                (pcr_df['testing_level'] == 'die') &
                                (pcr_df['hole'] == 0), 'timing'].dropna()

stat_pcr2_nacl_die = pcr_df.loc[(pcr_df['die'] == 3) &
                                (pcr_df['testing_day'] == 'd0') &
                                (pcr_df['liquid'] == 'NaCl 0.9%') &
                                (pcr_df['testing_level'] == 'die') &
                                (pcr_df['hole'] == 0), 'timing'].dropna()

data = [stat_pcr1_na_wafer.describe(), stat_pcr2_na_wafer.describe(),
        stat_pcr1_nacl_wafer.describe(), stat_pcr2_nacl_wafer.describe(), stat_pcr1_na_die.describe(),
        stat_pcr2_na_die.describe(), stat_pcr1_nacl_die.describe(), stat_pcr2_nacl_die.describe()]
var = [statistics.variance(stat_pcr1_na_wafer), statistics.variance(stat_pcr2_na_wafer),
       statistics.variance(stat_pcr1_nacl_wafer), statistics.variance(stat_pcr2_nacl_wafer),
       statistics.variance(stat_pcr1_na_die), statistics.variance(stat_pcr2_na_die),
       statistics.variance(stat_pcr1_nacl_die), statistics.variance(stat_pcr2_nacl_die)]
stat_df_2 = pd.DataFrame(data=pd.concat(data, axis=1))
stat_df_2 = stat_df_2.T
levels = ['wafer', 'wafer', 'wafer', 'wafer', 'die', 'die', 'die', 'die']
dies = [2, 3, 2, 3, 2, 3, 2, 3]
liquids = ['NA', 'NA', 'NaCl', 'NaCl', 'NA', 'NA', 'NaCl', 'NaCl']
stat_df_2['level'] = levels
stat_df_2['die'] = dies
stat_df_2['liquid'] = liquids
stat_df_2.set_index('level', inplace=True)
stat_df_2.set_index('die', append=True, inplace=True)
stat_df_2.set_index('liquid', append=True, inplace=True)
stat_df_2.insert(3, 'std_%', stat_df_2['std'] / stat_df_2['mean'])
stat_df_2.insert(4, 'var', var)

def t_value(sample1, sample2):
    t = (sample1.mean() - sample2.mean()) / \
        ((statistics.variance(sample1)**2 / sample1.count()) + (statistics.variance(sample2)**2 / sample2.count()))
    return t

def t_value2(sample1, sample2):
    t = (sample1.mean() - sample2.mean()) / \
        np.sqrt((statistics.variance(sample1) / sample1.count()) + (statistics.variance(sample2) / sample2.count()))
    return t

PCR2_wafer = pcr_df.loc[(pcr_df['die'] == 3) &
                        (pcr_df['testing_day'] == 'd0') &
                        (pcr_df['liquid'] == 'NaCl 0.9%') &
                        (pcr_df['testing_level'] == 'wafer') &
                        (pcr_df['hole'] == 0), 'timing'].dropna()

PCR2_die =  pcr_df.loc[(pcr_df['die'] == 3) &
                        (pcr_df['testing_day'] == 'd0') &
                        (pcr_df['liquid'] == 'NaCl 0.9%') &
                        (pcr_df['testing_level'] == 'die') &
                        (pcr_df['hole'] == 0), 'timing'].dropna()


# 3test
PCR2_die_nacl_wafer7 = pcr_df.loc[(pcr_df['die'] == 3) &
                                    (pcr_df['testing_day'] == 'd0') &
                                    (pcr_df['liquid'] == 'NaCl 0.9%') &
                                    (pcr_df['testing_level'] == 'die') &
                                    (pcr_df['wafer_number'] == 7) &
                                    (pcr_df['hole'] == 0), 'timing'].dropna()

PCR2_die_nacl_wafer9 = pcr_df.loc[(pcr_df['die'] == 3) &
                                    (pcr_df['testing_day'] == 'd0') &
                                    (pcr_df['liquid'] == 'NaCl 0.9%') &
                                    (pcr_df['testing_level'] == 'die') &
                                    (pcr_df['wafer_number'] == 9) &
                                    (pcr_df['hole'] == 0), 'timing'].dropna()

PCR2_die_nacl_wafer10 = pcr_df.loc[(pcr_df['die'] == 3) &
                                    (pcr_df['testing_day'] == 'd0') &
                                    (pcr_df['liquid'] == 'NaCl 0.9%') &
                                    (pcr_df['testing_level'] == 'die') &
                                    (pcr_df['wafer_number'] == 10) &
                                    (pcr_df['hole'] == 0), 'timing'].dropna()


combined = pd.concat([PCR2_die_nacl_wafer9, PCR2_die_nacl_wafer10])

sp.stats.ttest_ind(PCR2_die_nacl_wafer7, combined, equal_var=False)

sp.stats.f_oneway(PCR2_die_nacl_wafer7, PCR2_die_nacl_wafer9, PCR2_die_nacl_wafer10)

# Bubble stats
bubs_inlet_PCR1 = pcr_df.loc[
                            (pcr_df['testing_day'] == 'd0') &
                            (pcr_df['hole'] == 0) &
                            (pcr_df['bubble_inlet'].isin({0, 1})) &
                            (pcr_df['die'] == 2), 'bubble_inlet']
bubs_inlet_PCR2 = pcr_df.loc[
                                   (pcr_df['testing_day'] == 'd0') &
                                   (pcr_df['hole'] == 0) &
                                   (pcr_df['bubble_inlet'].isin({0, 1})) &
                                   (pcr_df['die'] == 3), 'bubble_inlet']


# Stats per wafer
pcr_df.loc[(pcr_df['die'] == 2) &
                                (pcr_df['testing_day'] == 'd0') &
                                (pcr_df['liquid'] == 'NA buffer') &
                                (pcr_df['testing_level'] == 'wafer') &
                                (pcr_df['hole'] == 0)].groupby('wafer')['timing'].describe()


# Graph level subplots
for level in {'wafer', 'die'}:
    sns.set_style('darkgrid')
    fig, axs = plt.subplots(1, 2, sharey=True)

    sns.boxplot(ax=axs[0], x="liquid", y="timing",
                      data=pcr_df.loc[(pcr_df['die'] == 2) & (pcr_df['testing_level'] == level) & (pcr_df['hole'] == 0)],
                hue='wafer', showfliers=False, showmeans=True)
    sns.stripplot(ax=axs[0], y='timing', x='liquid',
                  data=pcr_df.loc[(pcr_df['die'] == 2) & (pcr_df['testing_level'] == level) & (pcr_df['hole'] == 0)],
                  jitter=True,
                  dodge=True,
                  marker='o',
                  alpha=1,
                  hue='wafer',
                  palette='bright',
                  label=None)

    axs[0].get_legend().remove()

    boxes = sns.boxplot(ax=axs[1], x="liquid", y="timing",
                      data=pcr_df.loc[(pcr_df['die'] == 3) & (pcr_df['testing_level'] == level) & (pcr_df['hole'] == 0)],
                      hue='wafer', showfliers=False, showmeans=True)
    points = sns.stripplot(ax=axs[1], y='timing', x='liquid',
                  data=pcr_df.loc[(pcr_df['die'] == 3) & (pcr_df['testing_level'] == level) & (pcr_df['hole'] == 0)],
                  jitter=True,
                  dodge=True,
                  marker='o',
                  alpha=1,
                  hue='wafer',
                  palette='bright',
                  label='_nolegend_')

    wafers = pcr_df.loc[(pcr_df['die'] == 3) & (pcr_df['testing_level'] == level) & (pcr_df['hole'] == 0), 'wafer'].unique()

    legend_elements = [Patch(facecolor='blue', edgecolor='k',
                             label=wafers[0]),
                       Patch(facecolor='orange', edgecolor='k',
                             label=wafers[1]),
                       Patch(facecolor='green', edgecolor='k',
                             label=wafers[2])]
    axs[1].legend(handles=legend_elements, loc='upper right', title='Wafer')

    axs[0].set_title('PCR 1')
    axs[1].set_title('PCR 2')

    for ax in fig.get_axes():
        ax.grid(b=True, axis='y')
        ax.set_xlabel(None)
    fig.suptitle(f'Time in reactor distributions \n {level} level - day 0')
    axs[0].set_ylabel('Time (s)')
    axs[1].set_ylabel(None)

    plt.show()


# Graph PCR subplots
for die in {2, 3}:
    sns.set_style('darkgrid')
    fig, axs = plt.subplots(1, 2, sharey=True)

    sns.boxplot(ax=axs[0], x="liquid", y="timing",
                      data=pcr_df.loc[(pcr_df['die'] == die) & (pcr_df['testing_level'] == 'die') & (pcr_df['hole'] == 0)],
                hue='wafer', showfliers=False, showmeans=True)
    sns.stripplot(ax=axs[0], y='timing', x='liquid',
                  data=pcr_df.loc[(pcr_df['die'] == die) & (pcr_df['testing_level'] == 'die') & (pcr_df['hole'] == 0)],
                  jitter=True,
                  dodge=True,
                  marker='o',
                  alpha=1,
                  hue='wafer',
                  palette='bright',
                  label=None)

    wafers = pcr_df.loc[(pcr_df['die'] == 3) & (pcr_df['testing_level'] == 'die') & (pcr_df['hole'] == 0), 'wafer'].unique()

    legend_elements = [Patch(facecolor='blue', edgecolor='k',
                             label=wafers[0]),
                       Patch(facecolor='orange', edgecolor='k',
                             label=wafers[1]),
                       Patch(facecolor='green', edgecolor='k',
                             label=wafers[2])]
    axs[0].legend(handles=legend_elements, loc='upper right', title='Wafer')

    boxes = sns.boxplot(ax=axs[1], x="liquid", y="timing",
                      data=pcr_df.loc[(pcr_df['die'] == die) & (pcr_df['testing_level'] == 'wafer') & (pcr_df['hole'] == 0)],
                      hue='wafer', showfliers=False, showmeans=True)
    points = sns.stripplot(ax=axs[1], y='timing', x='liquid',
                  data=pcr_df.loc[(pcr_df['die'] == die) & (pcr_df['testing_level'] == 'wafer') & (pcr_df['hole'] == 0)],
                  jitter=True,
                  dodge=True,
                  marker='o',
                  alpha=1,
                  hue='wafer',
                  palette='bright',
                  label='_nolegend_')

    wafers = pcr_df.loc[(pcr_df['die'] == 3) & (pcr_df['testing_level'] == 'wafer') & (pcr_df['hole'] == 0), 'wafer'].unique()

    legend_elements = [Patch(facecolor='blue', edgecolor='k',
                             label=wafers[0]),
                       Patch(facecolor='orange', edgecolor='k',
                             label=wafers[1]),
                       Patch(facecolor='green', edgecolor='k',
                             label=wafers[2])]
    axs[1].legend(handles=legend_elements, loc='upper right', title='Wafer')

    axs[0].set_title('Die level')
    axs[1].set_title('Wafer level')

    for ax in fig.get_axes():
        ax.grid(b=True, axis='y')
        ax.set_xlabel(None)
    fig.suptitle(f'Time in reactor distributions \n PCR {die-1} - day 0')
    axs[0].set_ylabel('Time (s)')
    axs[1].set_ylabel(None)

    axs[0].set_ylim([0, 40])

    plt.show()


def plot_stackedbar_p(df_1, df_2, labels, colors, title):

    fields_1 = df_1.columns[2:4].tolist()
    fields_2 = df_2.columns[2:4].tolist()

    # figure and axis
    fig, axs = plt.subplots(1, 2, figsize=(12, 10), sharex=True, sharey=True)
    plt.subplots_adjust(left=0.2)

    # plot bars
    left = len(df_1) * [0]
    for idx, name in enumerate(fields_1):
        axs[0].barh(df_1.index, df_1[name], left=left, color=colors[idx])
        left = left + df_1[name]

    left = len(df_2) * [0]
    for idx, name in enumerate(fields_2):
        axs[1].barh(df_2.index, df_2[name], left=left, color=colors[idx])
        left = left + df_2[name]

    # title and subtitle
    fig.suptitle(title)
    #plt.text(0, ax.get_yticks()[-1] + 0.75, subtitle)

    # legend
    box = ([0.58, 0.93, 0, 0])
    fig.legend(labels, bbox_to_anchor=box, ncol=2, frameon=False)

    # remove spines
    #for ax in axs:
    #    ax.spines['right'].set_visible(False)
    #    ax.spines['left'].set_visible(False)
    #    ax.spines['top'].set_visible(False)
    #    ax.spines['bottom'].set_visible(False)

    # format x ticks
    xticks = np.arange(0, 1.1, 0.1)
    xlabels = ['{}%'.format(i) for i in np.arange(0, 101, 10)]
    plt.xticks(xticks, xlabels)

    # adjust limits and draw grid lines
    plt.ylim(-0.5, axs[0].get_yticks()[-1] + 0.5)
    axs[0].xaxis.grid(color='gray', linestyle='dashed')
    axs[1].xaxis.grid(color='gray', linestyle='dashed')

    axs[0].set_title('Die level')
    axs[1].set_title('Wafer level')

    axes2 = axs[0].twinx()  # mirror them
    ymin = axs[0].viewLim.extents[1]
    ymax = axs[0].viewLim.extents[3]
    axes2.set_ylim([ymin, ymax])

    ylabels = (df_1['Bubble'] + df_1['No_bubble']).values
    ylabels = [''] + ylabels.tolist() + ['']
    axes2.set_yticklabels(ylabels)

    secax_y2 = axs[0].secondary_yaxis(location=-0.2)
    secax_y2.set_ylim([ymin, ymax])
    y_loclabels = df_1.index
    y_loclabels = [''] + y_loclabels.tolist() + ['']
    secax_y2.set_yticklabels(y_loclabels)

    axes4 = axs[1].twinx()  # mirror them
    axes4.set_ylim([ymin, ymax])
    ylabels = (df_2['Bubble'] + df_2['No_bubble']).values
    ylabels = [''] + ylabels.tolist() + ['']
    axes4.set_yticklabels(ylabels)



for die in pcr_df['die'].unique():
    for level in pcr_df['testing_level'].unique():
        # variables
        pcr = die - 1
        title = f'Bubbles in ARAX system, PCR {pcr}\n'
        labels = ['Bubble', 'No bubble']
        colors = ['maroon', 'green']

        splitter_1 = pcr_df.loc[(pcr_df['bubble_splitter'] == 1) & (pcr_df['die'] == die) & (pcr_df['testing_level'] == level)].shape[0]
        splitter_0 = pcr_df.loc[(pcr_df['bubble_splitter'] == 0) & (pcr_df['die'] == die) & (pcr_df['testing_level'] == level)].shape[0]

        inlet_1 = pcr_df.loc[(pcr_df['bubble_inlet'] == 1) & (pcr_df['die'] == die) & (pcr_df['testing_level'] == level)].shape[0]
        inlet_0 = pcr_df.loc[(pcr_df['bubble_inlet'] == 0) & (pcr_df['die'] == die) & (pcr_df['testing_level'] == level)].shape[0]

        pre_1 = pcr_df.loc[(pcr_df['bubble_pre'] == 1) & (pcr_df['die'] == die) & (pcr_df['testing_level'] == level)].shape[0]
        pre_0 = pcr_df.loc[(pcr_df['bubble_pre'] == 0) & (pcr_df['die'] == die) & (pcr_df['testing_level'] == level)].shape[0]

        react_1 = pcr_df.loc[(pcr_df['bubble_reactor'] == 1) & (pcr_df['die'] == die) & (pcr_df['testing_level'] == level)].shape[0]
        react_0 = pcr_df.loc[(pcr_df['bubble_reactor'] == 0) & (pcr_df['die'] == die) & (pcr_df['testing_level'] == level)].shape[0]

        post_1 = pcr_df.loc[(pcr_df['bubble_post'] == 1) & (pcr_df['die'] == die) & (pcr_df['testing_level'] == level)].shape[0]
        post_0 = pcr_df.loc[(pcr_df['bubble_post'] == 0) & (pcr_df['die'] == die) & (pcr_df['testing_level'] == level)].shape[0]

        outlet_1 = pcr_df.loc[(pcr_df['bubble_outlet'] == 1) & (pcr_df['die'] == die) & (pcr_df['testing_level'] == level)].shape[0]
        outlet_0 = pcr_df.loc[(pcr_df['bubble_outlet'] == 0) & (pcr_df['die'] == die) & (pcr_df['testing_level'] == level)].shape[0]

        if level == 'die':
            if die == 3:
                die_bub_df = pd.DataFrame(data={'Inlet': [inlet_1, inlet_0],
                                            'Splitter': [splitter_1, splitter_0],
                                            'Pre-reactor': [pre_1, pre_0],
                                            'Reactor': [react_1, react_0],
                                            'Post-reactor': [post_1, post_0],
                                            'Outlet': [outlet_1, outlet_0]}).T
            elif die == 2:
                die_bub_df = pd.DataFrame(data={'Inlet': [inlet_1, inlet_0],
                                                'Pre-reactor': [pre_1, pre_0],
                                                'Reactor': [react_1, react_0],
                                                'Post-reactor': [post_1, post_0],
                                                'Outlet': [outlet_1, outlet_0]}).T
            die_bub_df.columns = ['Bubble', 'No_bubble']
            die_bub_df['Bubble_%'] = die_bub_df['Bubble'] / (die_bub_df['Bubble'] + die_bub_df['No_bubble'])
            die_bub_df['No_bubble_%'] = die_bub_df['No_bubble'] / (
                        die_bub_df['Bubble'] + die_bub_df['No_bubble'])

        if level == 'wafer':
            if die == 3:
                wafer_bub_df = pd.DataFrame(data={'Inlet': [inlet_1, inlet_0],
                                            'Splitter': [splitter_1, splitter_0],
                                            'Pre-reactor': [pre_1, pre_0],
                                            'Reactor': [react_1, react_0],
                                            'Post-reactor': [post_1, post_0],
                                            'Outlet': [outlet_1, outlet_0]}).T
            elif die == 2:
                wafer_bub_df = pd.DataFrame(data={'Inlet': [inlet_1, inlet_0],
                                                'Pre-reactor': [pre_1, pre_0],
                                                'Reactor': [react_1, react_0],
                                                'Post-reactor': [post_1, post_0],
                                                'Outlet': [outlet_1, outlet_0]}).T
            wafer_bub_df.columns = ['Bubble', 'No_bubble']
            wafer_bub_df['Bubble_%'] = wafer_bub_df['Bubble'] / (wafer_bub_df['Bubble'] + wafer_bub_df['No_bubble'])
            wafer_bub_df['No_bubble_%'] = wafer_bub_df['No_bubble'] / (wafer_bub_df['Bubble'] + wafer_bub_df['No_bubble'])

    plot_stackedbar_p(df_1=die_bub_df,
                      df_2=wafer_bub_df,
                      labels=labels, colors=colors, title=title)
