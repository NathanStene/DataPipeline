import tkinter as tk
from tkinter import filedialog
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

root = tk.Tk()

canvas1 = tk.Canvas(root, width=300, height=300, bg='lightsteelblue')
canvas1.pack()

def getExcel():
    global df_0, df_1

    import_file_path = filedialog.askopenfilename()
    df_0 = pd.read_excel(io=import_file_path, sheet_name=0, header=1)
    df_1 = pd.read_excel(io=import_file_path, sheet_name=1)
    print(df_0, df_1)
    return


browseButton_Excel = tk.Button(root, text='Import Excel File', command=getExcel, bg='green', fg='white',
                               font=('helvetica', 12, 'bold'))
canvas1.create_window(150, 150, window=browseButton_Excel)

root.mainloop()

# -------------

df = pd.DataFrame(data={'pcr': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                        'magnet': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        'swab': [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        'timing': [600, 1200, 603, 600, 330, 263, 245, 301, 216, 241, 283, 311, 261, 290, 482, 214, 543,
                                 246, 263, 452, 293, 297, 390, 393, 491, 563, 417, 472, 461, 741, 327, 359, 715]})
df['flow_speed'] = 50*60/df['timing']


for qty in {'timing', 'flow_speed'}:
    fig, axs = plt.subplots(1, 3, sharey=True)
    sns.set(style="whitegrid")

    data_1 = df.loc[(df['pcr'] == 1) & (df['magnet'] == 0)]
    data_3 = df.loc[(df['pcr'] == 1) & (df['magnet'] == 1)]
    data_4 = df.loc[(df['pcr'] == 2) & (df['magnet'] == 1)]

    ax = sns.boxplot(ax=axs[0], x="swab", y=qty,
                      data=data_1,
                      showfliers=False,
                showmeans=True)
    sns.swarmplot(ax=axs[0], x="swab", y=qty,
                        data=data_1,
                        color='blue')


    sns.boxplot(ax=axs[1], x="swab", y=qty,
                      data=data_3,
                      showfliers=False,
                showmeans=True)
    sns.swarmplot(ax=axs[1], x="swab", y=qty,
                        data=data_3,
                        color='blue')

    sns.boxplot(ax=axs[2], x="swab", y=qty,
                      data=data_4,
                      showfliers=False,
                showmeans=True)
    sns.swarmplot(ax=axs[2], x="swab", y=qty,
                        data=data_4,
                        color='blue')

    axs[0].set_title('PCR 1 \n No magnet')
    axs[1].set_title('PCR 1 \n Magnet')
    axs[2].set_title('PCR 2 \n Magnet')
    if qty == 'timing':
        fig.suptitle('Time in reactor distributions')
        axs[0].set_ylim([0, 1300])
        axs[0].set_ylabel('Time (s)')
        axs[1].set_ylabel(None)
        axs[2].set_ylabel(None)
    if qty == 'flow_speed':
        fig.suptitle('Flow speed in reactor distributions')
        axs[0].set_ylim([0, 15])
        axs[0].set_ylabel('Flow speed (\u03BCl/min)')
        axs[1].set_ylabel(None)
        axs[2].set_ylabel(None)

    axs[0].grid(b=True, axis='both')
    axs[1].grid(b=True, axis='both')
    axs[2].grid(b=True, axis='both')

plt.show()