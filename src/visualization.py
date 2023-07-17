# import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def score_bar_plot(df: pd.DataFrame) -> None:
    """
    Bar plot of scores by grading metric
    :param df: Input a copy of the original training data as a dataframe
    """
    # Count scores per grading criteria
    df.drop(columns=['text_id', 'full_text'], inplace=True)
    df = df.apply(pd.Series.value_counts)
    df = df.reset_index().rename(columns={'index': 'score'})
    df = pd.melt(df, id_vars=['score'], var_name='grading_metric', value_name='count')
    plt.figure()
    sns.barplot(x='score', y='count', hue='grading_metric', data=df)
    del df
    return


def avg_score_hist(df: pd.DataFrame, *,
                   bin_width: float = 0.25) -> None:
    """
    Histogram of a student's avg grading metric scores

    :param df: Input a copy of the original training data as a dataframe
    :param bin_width: Width of bins in the histogram
    """
    # Histogram of average grading metrics
    df.drop(columns=['text_id', 'full_text'], inplace=True)
    df['avg_score'] = df.mean(axis=1)
    fig = plt.figure()
    ax = sns.histplot(x='avg_score', data=df, binwidth=bin_width)
    del df, ax, fig
    return
