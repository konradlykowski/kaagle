import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display


def show_correlation_matrix(corr_matrix):
    plt.figure(figsize=(10, 10))
    plt.imshow(corr_matrix, cmap='RdYlGn', interpolation='none', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation='vertical')
    plt.yticks(range(len(corr_matrix)), corr_matrix.columns);
    plt.suptitle('Correlations Heat Map', fontsize=15, fontweight='bold')
    plt.show()


def show_violin_plot(data_frame, feature):
    fig, ax = plt.subplots()
    ax.violinplot(data_frame[feature], vert=False)
    plt.suptitle(feature, fontsize=15, fontweight='bold')
    plt.show()


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        display(df)
