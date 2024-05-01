import plotly.express as px


def scale_dataframe(df, reverse=False, min_values=None, max_values=None):
    if not reverse:
        min_values = df.min()
        max_values = df.max()

        range_values = max_values - min_values
        range_values[range_values == 0] = 1

        scaled_df = (df - min_values) / range_values

        return scaled_df, min_values, max_values
    else:
        if min_values is None or max_values is None:
            raise ValueError("min_values and max_values must be provided to reverse scaling.")

        range_values = max_values - min_values
        range_values[range_values == 0] = 1

        original_df = df * range_values + min_values
        return original_df


def visualize_df(df, columns, outcome_col, figure_path):
    gg = df[columns].drop_duplicates().reset_index().astype(float).drop(columns=['index'])

    fig = px.parallel_coordinates(
        gg,
        color=outcome_col,
        labels={e: e for e in columns},
        color_continuous_scale=px.colors.diverging.Tealrose,
        color_continuous_midpoint=gg[outcome_col].max() / 2)

    fig.update_layout(coloraxis_showscale=True)

    fig.write_image(figure_path)
