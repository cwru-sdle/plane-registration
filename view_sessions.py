import marimo

__generated_with = "0.13.14"
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
    import datetime
    import glob
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Try one of these:
    pio.renderers.default = 'notebook'        # For classic Jupyter Notebook
    return datetime, mo, np, os, pl, plt, px, sns


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
    mo.md(f"""**Current Settings:** Layers = {'ON' if array.value[2] else 'OFF'}""")
    return


@app.cell
def _(array, files, parquet_path, pl, px, summary_df):
    # Even better - do everything lazily:
    def create_3d_scatter(df, max_rows=10_000):
        row_count = df.select(pl.len()).collect().item()
        if row_count > max_rows:
            sample_fraction = max(1, int(row_count / max_rows))
            df = df.with_row_index("row_nr").filter(pl.col("row_nr") % sample_fraction == 0)

        # Compute outlier bounds lazily
        bounds = df.select([
            pl.col('layer').quantile(0.25).alias('q1'),
            pl.col('layer').quantile(0.75).alias('q3')
        ]).collect()
        q1, q3 = bounds[0, 'q1'], bounds[0, 'q3']
        iqr = q3 - q1
        lower_bound = q1 - 3*iqr
        upper_bound = q3 + 3*iqr

        # Final collection with all filters applied
        df_final = df.filter(
            (pl.col('layer') >= lower_bound) & (pl.col('layer') <= upper_bound)
        ).collect()

        if array.value[2]:
            # Use existing layer column with alternating red/black
            df_final = df_final.with_columns([
                (pl.col('layer') // array.value[1] % 2).alias('layer_color')
            ])
            fig = px.scatter_3d(
                df_final.to_pandas(), 
                x='x', y='y', z='layer',
                color='layer_color',
                color_discrete_map={0: 'red', 1: 'black'},
                opacity=0.5,
            )
        else:
            # Original sensor1 coloring
            fig = px.scatter_3d(
                df_final.to_pandas(), 
                x='x', y='y', z='layer',
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
            aspectratio=dict(x=1,y=1,z=1)),
            scene_dragmode='turntable')
        return fig, df_final
    selected_index = files.index(array.value[0])
    filename = summary_df['filename'][selected_index]
    save_filename = parquet_path+filename.replace(".parquet","")+".txt"
    df = pl.scan_parquet(parquet_path+filename)
    fig, df_final = create_3d_scatter(df)
    return df, df_final, fig, save_filename


@app.cell
def _(array, df_final, fig):
    fig.show()
    print(array.value[0])
    print(df_final.height)
    print("Unique layers:", df_final['layer'].unique_counts().len())
    return


@app.cell
def _(df, np, pl):
    # Define histogram bin edges
    x_max = df.select(pl.col("x").max()).collect().item()
    x_min = df.select(pl.col("x").min()).collect().item()
    y_max = df.select(pl.col("y").max()).collect().item()
    y_min = df.select(pl.col("y").min()).collect().item()

    bins = 1000

    x_edges = np.linspace(x_min, x_max, bins + 1)
    y_edges = np.linspace(y_min, y_max, bins + 1)

    # Initialize histogram accumulator
    hist = np.zeros((bins, bins), dtype=np.int64)


    # Process in batches
    stream_df = df.select(["x", "y"])

    # Materialize in small batches
    for batch in stream_df.collect(engine="streaming").iter_slices(n_rows=100_000):
        x = batch["x"].to_numpy()
        y = batch["y"].to_numpy()
        h, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
        hist += h.astype(np.int64)
    return (hist,)


@app.cell
def _(hist, np, plt, sns):
    plt.figure(figsize=(8, 6))
    sns.heatmap(hist.T, cmap="viridis", cbar=True,vmin=0, vmax=np.percentile(hist.T, 98))
    plt.title("Heatmap of X-Y positions")
    plt.xlabel("X bins")
    plt.ylabel("Y bins")
    plt.gca().invert_yaxis()
    plt.show()
    return


@app.cell
def _(save_filename):
    open(save_filename, "a").close()

    # Read and print the contents
    with open(save_filename, "r") as f2:
        content = f2.read()
        print(f"File :{save_filename}:")
        print('='*100)
        print(content)
        f2.flush()
    return


@app.cell
def _(mo):
    # Create the UI elements
    text_input = mo.ui.text(placeholder="Enter your notes here...")
    save_button = mo.ui.run_button(label="Save Notes", kind="success")

    # Display them
    mo.hstack([text_input, save_button])
    return save_button, text_input


@app.cell
def _(datetime, save_button, save_filename, text_input):
    # Check if button was clicked and save the text
    if save_button.value:
        notes_text = text_input.value

        # Save to file
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(save_filename, "a") as f1:
            f1.write(f"\nTimestamp: {timestamp}\n")
            f1.write("-"*40 + "\n")
            f1.write(notes_text+"\n")
            f1.flush()
        print(f"✅ Notes saved to: {save_filename}")
        print(f"📝 Content: {notes_text}")

    return


@app.cell
def _(save_button, save_filename):
    if save_button.value:
        open(save_filename, "a").close()

        # Read and print the contents
        with open(save_filename, "r") as f3:
            content1 = f3.read()
            print(f"File :{save_filename}:")
            print('='*100)
            print(content1)
            f3.flush()
    return


if __name__ == "__main__":
    app.run()
