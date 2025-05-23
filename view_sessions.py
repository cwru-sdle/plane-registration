import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pyarrow
    import polars as pl
    import os
    return mo, os, pl


@app.cell
def _(os):
    parquet_path = "/mnt/vstor/CSE_MSE_RXF131/lab-staging/mds3/AdvManu/aconity_parquet/"
    files = os.listdir(parquet_path)
    return files, parquet_path


@app.cell
def _(files, mo):
    selected = mo.ui.dropdown(options = files, label ="Select Session")
    selected
    return (selected,)


@app.cell
def _(parquet_path, pl, selected):
    file_path = parquet_path + selected.value
    df = pl.read_parquet(file_path)
    print(df.columns)
    return (df,)


@app.cell
def _(df):
    import plotly.express as px

    # Create interactive 3D scatter plot
    fig = px.scatter_3d(df, x='x', y='y', z='z',
                        color='sensor1',
                        color_continuous_scale='Viridis',
                        size_max=5,
                        opacity=0.7,
                        title='3D Scatter: x, y, z colored by sensor1')

    fig.show()

    return


if __name__ == "__main__":
    app.run()
