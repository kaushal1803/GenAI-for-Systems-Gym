import pandas as pd
import re
import matplotlib.pyplot as plt

def rm_suf(col):
    return re.sub(r'_\d+B$', '', col)

def plot_task1():
    df = pd.read_csv('task-logs/task1.mpki.csv')

    df['bm'] = df['bm'].apply(rm_suf)
    df.loc[df['bm'] == 'libquantum', 'bm'] = 'libq'
    df.loc[df['bm'] == 'leslie3d', 'bm'] = 'leslie'
    df.loc[df['bm'] == 'GemsFDTD', 'bm'] = 'gems'
    df.loc[df['bm'] == 'xalancbmk', 'bm'] = 'xalanc'
    
    plt.bar(df['bm'], df['mpki'], color='skyblue')
    plt.ylabel('MPKI')
    plt.xticks(rotation=45)
    plt.yticks(range(0, int(df['mpki'].max()) + 10, 10))
    
    plt.tight_layout()
    #plt.show()
    plt.savefig('figures/task1.png')

plot_task1()

