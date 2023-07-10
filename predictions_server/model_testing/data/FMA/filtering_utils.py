import pandas as pd
import plotly.express as px
import os

DS_PATH = os.path.join(os.getcwd() + '/fma_metadata/')


def get_value_counts(df: pd.DataFrame, col_name: str | tuple[str, ...], min_occurrences: int = 0) -> dict[str | int | float, int]:
    value_counts = {}
    for value, occurrences in df[col_name].value_counts().items():
        if occurrences > min_occurrences:
            value_counts[value] = occurrences
    print(len(value_counts), 'values left')
    return pd.Series(value_counts)


def draw_pie(df: pd.DataFrame, col_name: str, min_occurrences: int = 0) -> None:
    value_counts = get_value_counts(df, col_name, min_occurrences)
    fig = px.pie(df[col_name], values=value_counts.values, names=value_counts.index)
    fig.update_traces(hoverinfo='label+percent', textinfo='percent')
    fig.show()
