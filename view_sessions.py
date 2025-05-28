import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pyarrow
    import polars as pl
    import os
    import numpy as np
    import plotly.io as pio
    # Now plot with the sampled df
    import plotly.express as px

    # Try one of these:
    pio.renderers.default = 'notebook'        # For classic Jupyter Notebook
    return mo, os, pl, px


@app.cell
def _(os, pl):
    parquet_path = "/mnt/vstor/CSE_MSE_RXF131/lab-staging/mds3/AdvManu/aconity_parquet/"

    def _():
        files = os.listdir(parquet_path)
        file_info = []
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(parquet_path, file)
                try:
                    row_count = pl.scan_parquet(file_path).select(pl.len()).collect().item()
                    file_size = os.path.getsize(file_path)
                    file_info.append({
                        'filename': file,
                        'rows': row_count,
                        'size_mb': file_size / (1024 * 1024)
                    })
                except Exception as e:
                    file_info.append({
                        'filename': file,
                        'rows': 0,
                        'size_mb': 0,
                        'error': str(e)
                    })
        # Create DataFrame and display
        summary_df = pl.DataFrame(file_info)
        return summary_df
    def format_size(rows):
        rows = int(rows)
        if rows >= 1_000_000_000_000:
            return f"{rows // 1_000_000_000}T"
        elif rows >= 1_000_000_000:
            return f"{rows // 1_000_000_000}G"
        elif rows >= 1_000_000:
            return f"{rows // 1_000_000}M"
        elif rows >= 1_000:
            return f"{rows // 1000}K"
        else:
            return str(rows)
    # Assuming ().iterrows() gives (file, row, size)
    files = [
        f"{file[0:13]}-{format_size(row)}"
        for (file, row, size) in _().iter_rows()
    ]
    summary_df = _()
    return files, parquet_path, summary_df


@app.cell
def _(files, mo):
    array = mo.ui.array([
        mo.ui.dropdown(options=files, label="Select Session", value=files[0]),
        mo.ui.dropdown(options=[30, 60, 100], label="Layer Height (μm)", value=30),
        mo.ui.button(
            value=False, 
            on_click=lambda value: not value, 
            label="Toggle Layers", 
            kind="warn"
        )
    ])
    return (array,)


@app.cell
def _(array):
    array
    return


@app.cell
def _(array, mo):
    mo.md(f"**Current Settings:** Layers = {'ON' if array.value[2] else 'OFF'}")
    return


@app.cell
def _(array, files, parquet_path, pl, px, summary_df):
    # Even better - do everything lazily:
    def create_3d_scatter(df, max_rows=100_000):
        row_count = df.select(pl.len()).collect().item()
        if row_count > max_rows:
            sample_fraction = max(1, int(row_count / max_rows))
            df = df.with_row_index("row_nr").filter(pl.col("row_nr") % sample_fraction == 0)
    
        # Compute outlier bounds lazily
        bounds = df.select([
            pl.col('z').quantile(0.25).alias('q1'),
            pl.col('z').quantile(0.75).alias('q3')
        ]).collect()
        q1, q3 = bounds[0, 'q1'], bounds[0, 'q3']
        iqr = q3 - q1
        lower_bound = q1 - 3*iqr
        upper_bound = q3 + 3*iqr
    
        # Final collection with all filters applied
        df_final = df.filter(
            (pl.col('z') >= lower_bound) & (pl.col('z') <= upper_bound)
        ).collect()
    
        if array.value[2]:
            # Use existing layer column with alternating red/black
            df_final = df_final.with_columns([
                (pl.col('layer') // array.value[1] % 2).alias('layer_color')
            ])
            fig = px.scatter_3d(
                df_final.to_pandas(), 
                x='x', y='y', z='z',
                color='layer_color',
                color_discrete_map={0: 'red', 1: 'black'},
                opacity=0.5,
            )
        else:
            # Original sensor1 coloring
            fig = px.scatter_3d(
                df_final.to_pandas(), 
                x='x', y='y', z='z',
                color='sensor1',
                color_continuous_scale='Viridis',
                opacity=0.1,
            )
        fig.update_traces(marker=dict(size=1))  # Adjust size here
        fig.update_layout(
            scene=dict(
                aspectmode="manual",
                camera=dict(
                    eye=dict(x=0, y=0, z=2.5),    # Camera directly above, looking down
                    center=dict(x=0, y=0, z=0),   # Looking at the center
                    up=dict(x=0, y=1, z=0)       # X-axis is "up" (horizontal in view)
                ),
            aspectratio=dict(x=1,y=1,z=.1)),
            scene_dragmode='turntable')
        return fig, df_final
    selected_index = files.index(array.value[0])
    filename = summary_df['filename'][selected_index]
    df = pl.scan_parquet(parquet_path+filename)
    fig, df_final = create_3d_scatter(df)
    return df_final, fig


@app.cell
def _(array, df_final, fig):
    fig.show()
    print(array.value[0])
    print(df_final.height)
    print("Unique layers:", df_final['layer'].unique_counts().len())
    return


if __name__ == "__main__":
    app.run()
