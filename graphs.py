import os
import pandas as pd
import numpy as np
import math
from operator import add, truediv
from matplotlib.patches import Patch
from matplotlib import pyplot as plt


class Graphs:

    def __init__(self, pickle_name):
        working_dir = os.getcwd()
        self.pickle_name = pickle_name
        try:
            self.data = pd.read_pickle(os.path.join(working_dir, f'pickles\\{self.pickle_name}.pkl'))
        except OSError:
            print('Wow, you\'re going a little too fast my friend!\nTry saving the dataframe as a pickle first...')
        return

    def plot_bubbles_vertical(self, wafer_discriminator=False, normalized=True):
        pcr_df = self.data
        if pcr_df['testing_level'].unique().shape[0] > 1:
            levels = ['die', 'wafer']
        else:
            levels = ['die']

        for die in pcr_df['die'].unique():
            locations = []
            for column in pcr_df.columns:
                if column.split('_')[0] == 'bubble':
                    locations.append(column)

            location_dict = {'bubble_inlet': 'Inlet', 'bubble_pre': 'Pre-reactor', 'bubble_reactor': 'Reactor',
                             'bubble_post': 'Post-reactor', 'bubble_outlet': 'Outlet', 'bubble_splitter': 'Splitter'}

            if die == 2:
                locations.remove('bubble_splitter')

            x_labels = []
            for location in locations:
                x_labels.append(location_dict[location])

            if wafer_discriminator is False:
                width = 0.35
                bubs = []
                no_bubs = []
                for level in levels:
                    for loc in locations:
                        to_append_present = pcr_df.loc[(pcr_df[f'{loc}'] == 1) &
                                                (pcr_df['die'] == die) & (pcr_df['testing_level'] == level)].shape[0]
                        to_append_absent = pcr_df.loc[(pcr_df[f'{loc}'] == 0) &
                                                (pcr_df['die'] == die) & (pcr_df['testing_level'] == level)].shape[0]

                        bubs.append(to_append_present)
                        no_bubs.append(to_append_absent)

                bubs_die = bubs[0:int(len(bubs)/2)]
                bubs_wafer = bubs[int(len(bubs)/2):len(bubs)]

                no_bubs_die = no_bubs[0:int(len(no_bubs)/2)]
                no_bubs_wafer = no_bubs[int(len(no_bubs)/2):len(no_bubs)]

                total_die = list(map(add, bubs_die, no_bubs_die))
                total_wafer = list(map(add, bubs_wafer, no_bubs_wafer))

                def list_div(list1, list2):
                    result = []
                    for i in range(len(list1)):
                        if list2[i] == 0:
                            result.append(0)
                        else:
                            result.append(list1[i]/list2[i])
                    return result

                if normalized:
                    y_label = '%'
                    bubs_die = list_div(bubs_die, total_die)
                    bubs_die = [i * 100 for i in bubs_die]
                    bubs_wafer = list_div(bubs_wafer, total_wafer)
                    bubs_wafer = [i * 100 for i in bubs_wafer]
                    no_bubs_die = list_div(no_bubs_die, total_die)
                    no_bubs_die = [i * 100 for i in no_bubs_die]
                    no_bubs_wafer = list_div(no_bubs_wafer, total_wafer)
                    no_bubs_wafer = [i * 100 for i in no_bubs_wafer]
                    total_die_new = [100] * len(bubs_die)
                    total_wafer_new = [100] * len(bubs_die)
                else:
                    y_label = 'N samples'
                    total_die_new = total_die
                    total_wafer_new = total_wafer

                null_values_wafer = pd.Series([i == 0 for i in total_wafer])
                position_wafer = np.where(null_values_wafer)
                null_values_die = pd.Series([i == 0 for i in total_die])
                position_die = np.where(null_values_die)

                if null_values_wafer.any():
                    total_wafer_new[pd.Series(position_wafer)[0][0]] = 0
                if null_values_die.any():
                    total_die_new[pd.Series(position_die)[0][0]] = 0

                def annotate_graph(bar, level, total):
                    if level == 'die':
                        ax = axs[0]
                    elif level == 'wafer':
                        ax = axs[1]
                    i = 0
                    for rect in bar:
                        count = total[i]
                        annotation = '{}'.format(count)
                        height = 100
                        text = ax.annotate(annotation,
                                           xy=(rect.get_x() + rect.get_width() / 2, height),
                                           xytext=(0, 3),  # 3 points vertical offset
                                           textcoords="offset points",
                                           ha='center', va='bottom')
                        text.set_fontsize(7.5)
                        i = i + 1

                fig, axs = plt.subplots(1, len(levels), sharey=True, sharex=True)
                pcr = die - 1
                fig.suptitle(f'Bubbles in PCR {pcr}, D0')
                x = np.arange(len(x_labels))

                if len(levels) > 1:
                    bar1 = axs[0].bar(x, total_die_new, width, color='g')
                    axs[0].bar(x, bubs_die, width, color='r')

                    bar2 = axs[1].bar(x, total_wafer_new, width, color='g')
                    axs[1].bar(x, bubs_wafer, width, color='r')

                    axs[0].set_ylabel(y_label)
                    axs[0].set_title(f'Die level')
                    axs[1].set_title(f'Wafer level')
                    axs[0].set_xticks(x)
                    axs[0].set_xticklabels(x_labels)
                    if normalized:
                        axs[0].set_yticks(np.arange(0, 101, 10))
                        annotate_graph(bar1, level='die', total=total_die)
                        annotate_graph(bar2, level='wafer', total=total_wafer)

                    axs[0].grid(axis='y', linestyle='dashed')
                    axs[1].grid(axis='y', linestyle='dashed')
                elif len(levels) == 1:
                    level = levels[0]
                    if level == 'die':
                        total_new = total_die_new
                        total = total_die
                        bubs = bubs_die
                    elif level == 'wafer':
                        total_new = total_wafer_new
                        total = total_wafer
                        bubs = bubs_wafer

                    bar1 = axs.bar(x, total_new, width, color='g')
                    axs.bar(x, bubs, width, color='r')

                    axs.set_ylabel(y_label)
                    axs.set_title(f'{level} level')
                    axs.set_xticks(x)
                    axs.set_xticklabels(x_labels)
                    if normalized:
                        axs.set_yticks(np.arange(0, 101, 10))
                        annotate_graph(bar1, level=level, total=total)

                    axs.grid(axis='y', linestyle='dashed')

                plt.show()

            elif wafer_discriminator is True:
                width = 0.20
                bubs = pd.DataFrame()
                no_bubs = pd.DataFrame()
                ntests = pd.DataFrame()
                for level in levels:
                    bubs_half = pd.DataFrame(columns=locations,
                                             index=pcr_df.loc[pcr_df['testing_level'] == level, 'wafer_number'].unique())
                    no_bubs_half = pd.DataFrame(columns=locations,
                                                index=pcr_df.loc[pcr_df['testing_level'] == level, 'wafer_number'].unique())
                    ntests_half = pd.DataFrame(columns=locations,
                                                index=pcr_df.loc[pcr_df['testing_level'] == level, 'wafer_number'].unique())
                    for loc in locations:
                        to_append_present = pcr_df.loc[(pcr_df[f'{loc}'] == 1) &
                                                       (pcr_df['die'] == die) &
                                                       (pcr_df['testing_level'] == level) &
                                                       (pcr_df['testing_day'] == 'd0')].groupby('wafer_number').size()
                        to_append_absent = pcr_df.loc[(pcr_df[f'{loc}'] == 0) &
                                                      (pcr_df['die'] == die) &
                                                      (pcr_df['testing_level'] == level) &
                                                      (pcr_df['testing_day'] == 'd0')].groupby('wafer_number').size()
                        to_append_ntests = pcr_df.loc[(pcr_df['die'] == die) &
                                                  (pcr_df['testing_level'] == level) &
                                                  (pcr_df['testing_day'] == 'd0')].groupby('wafer_number').size()

                        bubs_half[f'{loc}'] = to_append_present
                        no_bubs_half[f'{loc}'] = to_append_absent
                        ntests_half[f'{loc}'] = to_append_ntests

                    bubs = bubs.append(bubs_half)
                    no_bubs = no_bubs.append(no_bubs_half)
                    ntests = ntests.append(ntests_half)

                bubs = bubs.fillna(0)
                no_bubs = no_bubs.fillna(0)

                bubs_die = bubs.iloc[0:int(bubs.shape[0]/2)]
                bubs_wafer = bubs[int(bubs.shape[0]/2):bubs.shape[0]]

                no_bubs_die = no_bubs[0:int(no_bubs.shape[0]/2)]
                no_bubs_wafer = no_bubs[int(no_bubs.shape[0]/2):no_bubs.shape[0]]

                total_die = bubs_die + no_bubs_die
                total_wafer = bubs_wafer + no_bubs_wafer

                def list_div(list1, list2):
                    result = []
                    for i in range(len(list1)):
                        if list2[i] == 0:
                            result.append(0)
                        else:
                            result.append(list1[i] / list2[i])
                    return result

                if normalized:
                    y_label = '%'
                    bubs_die = 100 * bubs_die / total_die
                    bubs_wafer = 100 * bubs_wafer / total_wafer
                    no_bubs_die = 100 * no_bubs_die / total_die
                    no_bubs_wafer = 100 * no_bubs_wafer / total_wafer
                    total_die_new = total_die.copy(deep=True)
                    total_wafer_new = total_wafer.copy(deep=True)
                    total_die_new.loc[:, :] = 100
                    total_wafer_new.loc[:, :] = 100
                else:
                    y_label = 'N samples'
                    total_die_new = total_die.copy(deep=True)
                    total_wafer_new = total_wafer.copy(deep=True)

                null_values_die = np.where(total_die == 0)
                null_values_wafer = np.where(total_wafer == 0)

                if len(null_values_wafer[0]) > 0:
                    for i in range(len(null_values_wafer[0])):
                        total_wafer_new.iloc[null_values_wafer[0][i], null_values_wafer[1][i]] = 0
                if len(null_values_die[0]) > 0:
                    for i in range(len(null_values_die[0])):
                        total_die_new.iloc[null_values_die[0][i], null_values_die[1][i]] = 0

                fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
                pcr = die - 1
                fig.suptitle(f'Bubbles in PCR {pcr}, D0')
                x = np.arange(len(x_labels))
                r2 = [i + width + 0.05 for i in x]
                r3 = [i + width + 0.05 for i in r2]

                # Subplot 1
                bar_tests1 = axs[0].bar(x, ntests.iloc[0, :], width, color='k')
                bar_tot1 = axs[0].bar(x, total_die_new.iloc[0, :], width, color='g')
                bar_bubs1 = axs[0].bar(x, bubs_die.iloc[0, :], width, color='r')

                bar_tests2 = axs[0].bar(r2, ntests.iloc[1, :], width, color='k')
                bar_tot2 = axs[0].bar(r2, total_die_new.iloc[1, :], width, color='g')
                bar_bubs2 = axs[0].bar(r2, bubs_die.iloc[1, :], width, color='r')

                bar_tests3 = axs[0].bar(r3, ntests.iloc[2, :], width, color='k')
                bar_tot3 = axs[0].bar(r3, total_die_new.iloc[2, :], width, color='g')
                bar_bubs3 = axs[0].bar(r3, bubs_die.iloc[2, :], width, color='r')

                # Subplot 2
                bar_tests4 = axs[1].bar(x, ntests.iloc[3, :], width, color='k')
                bar_tot4 = axs[1].bar(x, total_wafer_new.iloc[0, :], width, color='g')
                bar_bubs4 = axs[1].bar(x, bubs_wafer.iloc[0, :], width, color='r')

                bar_tests5 = axs[1].bar(r2, ntests.iloc[4, :], width, color='k')
                bar_tot5 = axs[1].bar(r2, total_wafer_new.iloc[1, :], width, color='g')
                bar_bubs5 = axs[1].bar(r2, bubs_wafer.iloc[1, :], width, color='r')

                bar_tests6 = axs[1].bar(r3, ntests.iloc[5, :], width, color='k')
                bar_tot6 = axs[1].bar(r3, total_wafer_new.iloc[2, :], width, color='g')
                bar_bubs6 = axs[1].bar(r3, bubs_wafer.iloc[2, :], width, color='r')

                axs[0].set_ylabel(y_label)
                axs[0].set_title(f'Die level')
                axs[1].set_title(f'Wafer level')
                axs[0].set_xticks(r2)
                axs[0].set_xticklabels(x_labels)
                if normalized:
                    axs[0].set_yticks(np.arange(0, 101, 10))
                else:
                    axs[0].set_yticks(np.arange(0, 22, 2))
                    legend_elements = [Patch(facecolor='k', edgecolor='k',
                                             label='Not measured'),
                                       Patch(facecolor='green', edgecolor='k',
                                             label='No bubble'),
                                       Patch(facecolor='red', edgecolor='k',
                                             label='Bubble')]
                    axs[0].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 0.))
                axs[0].grid(axis='y', linestyle='dashed')
                axs[1].grid(axis='y', linestyle='dashed')

                def annotate_graph(bar_tot, bar_bubs, bar_tests, level, total, wafer, normalized):
                    if level == 'die':
                        ax = axs[0]
                    elif level == 'wafer':
                        ax = axs[1]
                    i = 0
                    for rect in bar_tot:
                        count = total.iloc[wafer, :].values[i]
                        wafer_number = total.iloc[wafer, :].name
                        rect_height = rect.get_height()
                        if rect_height == 0:
                            perc = 0
                        else:
                            perc = round(100 * bar_bubs[i].get_height() / rect.get_height())
                        if normalized:
                            annotation = '{}\nD{}'.format(count.astype(int), wafer_number)
                            height = 100
                        else:
                            annotation = 'D{}\n{}%'.format(wafer_number, perc)
                            height = bar_tests[i].get_height()
                        text = ax.annotate(annotation,
                                    xy=(rect.get_x() + rect.get_width() / 2, height),
                                    xytext=(0, 3),  # 3 points vertical offset
                                    textcoords="offset points",
                                    ha='center', va='bottom')
                        text.set_fontsize(7.5)
                        i = i + 1

                annotate_graph(bar_tot1, bar_bubs1, bar_tests1, level='die', total=total_die, wafer=0, normalized=normalized)
                annotate_graph(bar_tot2, bar_bubs2, bar_tests2, level='die', total=total_die, wafer=1, normalized=normalized)
                annotate_graph(bar_tot3, bar_bubs3, bar_tests3, level='die', total=total_die, wafer=2, normalized=normalized)
                annotate_graph(bar_tot4, bar_bubs4, bar_tests4, level='wafer', total=total_wafer, wafer=0, normalized=normalized)
                annotate_graph(bar_tot5, bar_bubs5, bar_tests5, level='wafer', total=total_wafer, wafer=1, normalized=normalized)
                annotate_graph(bar_tot6, bar_bubs6, bar_tests6, level='wafer', total=total_wafer, wafer=2, normalized=normalized)

                plt.show()

    def plot_bubbles_horizontal(self):
        pcr_df = self.data

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
            # plt.text(0, ax.get_yticks()[-1] + 0.75, subtitle)

            # legend
            box = ([0.58, 0.93, 0, 0])
            fig.legend(labels, bbox_to_anchor=box, ncol=2, frameon=False)

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

                splitter_1 = pcr_df.loc[(pcr_df['bubble_splitter'] == 1) & (pcr_df['die'] == die) & (
                            pcr_df['testing_level'] == level)].shape[0]
                splitter_0 = pcr_df.loc[(pcr_df['bubble_splitter'] == 0) & (pcr_df['die'] == die) & (
                            pcr_df['testing_level'] == level)].shape[0]

                inlet_1 = pcr_df.loc[(pcr_df['bubble_inlet'] == 1) & (pcr_df['die'] == die) & (
                            pcr_df['testing_level'] == level)].shape[0]
                inlet_0 = pcr_df.loc[(pcr_df['bubble_inlet'] == 0) & (pcr_df['die'] == die) & (
                            pcr_df['testing_level'] == level)].shape[0]

                pre_1 = pcr_df.loc[(pcr_df['bubble_pre'] == 1) & (pcr_df['die'] == die) & (
                            pcr_df['testing_level'] == level)].shape[0]
                pre_0 = pcr_df.loc[(pcr_df['bubble_pre'] == 0) & (pcr_df['die'] == die) & (
                            pcr_df['testing_level'] == level)].shape[0]

                react_1 = pcr_df.loc[(pcr_df['bubble_reactor'] == 1) & (pcr_df['die'] == die) & (
                            pcr_df['testing_level'] == level)].shape[0]
                react_0 = pcr_df.loc[(pcr_df['bubble_reactor'] == 0) & (pcr_df['die'] == die) & (
                            pcr_df['testing_level'] == level)].shape[0]

                post_1 = pcr_df.loc[(pcr_df['bubble_post'] == 1) & (pcr_df['die'] == die) & (
                            pcr_df['testing_level'] == level)].shape[0]
                post_0 = pcr_df.loc[(pcr_df['bubble_post'] == 0) & (pcr_df['die'] == die) & (
                            pcr_df['testing_level'] == level)].shape[0]

                outlet_1 = pcr_df.loc[(pcr_df['bubble_outlet'] == 1) & (pcr_df['die'] == die) & (
                            pcr_df['testing_level'] == level)].shape[0]
                outlet_0 = pcr_df.loc[(pcr_df['bubble_outlet'] == 0) & (pcr_df['die'] == die) & (
                            pcr_df['testing_level'] == level)].shape[0]

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
                    wafer_bub_df['Bubble_%'] = wafer_bub_df['Bubble'] / (
                                wafer_bub_df['Bubble'] + wafer_bub_df['No_bubble'])
                    wafer_bub_df['No_bubble_%'] = wafer_bub_df['No_bubble'] / (
                                wafer_bub_df['Bubble'] + wafer_bub_df['No_bubble'])

            plot_stackedbar_p(df_1=die_bub_df,
                              df_2=wafer_bub_df,
                              labels=labels, colors=colors, title=title)


if __name__ == '__main__':
    sp = Graphs('pcr_df')
    sp.plot_bubbles_vertical(wafer_discriminator=True, normalized=False)

