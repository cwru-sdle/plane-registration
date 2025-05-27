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
    selected = mo.ui.dropdown(options=files, label="Select Session",value=files[0])
    selected
    return (selected,)


@app.cell
def _(files, parquet_path, pl, selected, summary_df):
    selected_index = files.index(selected.value)
    filename = summary_df['filename'][selected_index]
    df = pl.read_parquet(parquet_path+filename)
    return (df,)


@app.cell
def _(df, pl, px):
    def create_3d_scatter(df, max_rows=10_000):
        required_cols = ['x', 'y', 'z', 'sensor1']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # Compute IQR to remove z outliers
        z_stats = df.select([
            pl.col('z').quantile(0.25).alias('q1'),
            pl.col('z').quantile(0.75).alias('q3')
        ])
        q1, q3 = z_stats[0, 'q1'], z_stats[0, 'q3']
        iqr = q3 - q1
        lower_bound = q1 - 5 * iqr
        upper_bound = q3 + 5 * iqr

        # Filter out outliers in z
        df_filtered = df.filter((pl.col('z') >= lower_bound) & (pl.col('z') <= upper_bound))

        n = df_filtered.height
        if n > max_rows:
            print(f"Sampling {max_rows:,} points from {n:,} after outlier removal")
            df_sampled = df_filtered.sample(n=max_rows)
        else:
            df_sampled = df_filtered

        fig = px.scatter_3d(
            df_sampled.to_pandas(), 
            x='x', y='y', z='z',
            color='sensor1',
            color_continuous_scale='Viridis',
            size_max=5,
            opacity=0.7,
            title=f'3D Scatter: x, y, z colored by sensor1 (n={df_sampled.height:,})'
        )
        fig.update_layout(scene_dragmode='orbit')
        return fig
    fig = create_3d_scatter(df)
    return (fig,)


@app.cell
def _(fig, selected):
    fig.show()
    print(selected.value)
    return


if __name__ == "__main__":
    app.run()
