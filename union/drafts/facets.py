import os
import math
import pandas as pd
import gpxpy
import fit2gpx
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_activities(folder="./data/gpx", plot=True, return_data=True):
    """
    Analyze GPX and FIT files from a specified folder.
    
    Parameters:
    -----------
    folder : str, default "./data/gpx"
        Path to the folder containing GPX and FIT files
    plot : bool, default True
        Whether to create and display the visualization
    return_data : bool, default True
        Whether to return the processed dataframe
    
    Returns:
    --------
    pd.DataFrame or None
        Combined dataframe with all activities if return_data=True, else None
    """
    
    # Check if folder exists
    if not os.path.exists(folder):
        print(f"Error: Directory '{folder}' not found!")
        print(f"Current working directory: {os.getcwd()}")
        return None
    
    all_files = os.listdir(folder)
    gpx_fit_files = [os.path.join(folder, f) for f in all_files if f.endswith((".gpx", ".fit"))]
    
    if not gpx_fit_files:
        print(f"No GPX or FIT files found in '{folder}'")
        return None
    
    print(f"Found {len(gpx_fit_files)} files to process")
    
    dfs = []
    
    for filepath in gpx_fit_files:
        print(f"Processing: {filepath}")
        
        if filepath.endswith(".gpx"):
            with open(filepath, encoding="utf-8") as f:
                try:
                    activity = gpxpy.parse(f)
                except Exception as e:
                    print(f"Error in the file ({filepath}): {e}")
                    continue
            
            lon, lat, ele, time, name, dist = [], [], [], [], [], []
            
            for track in activity.tracks:
                for segment in track.segments:
                    if not segment.points:
                        continue
                    x0, y0, d0 = segment.points[0].longitude, segment.points[0].latitude, 0
                    for point in segment.points:
                        x, y, z, t = point.longitude, point.latitude, point.elevation, point.time
                        d = d0 + math.sqrt((x - x0) ** 2 + (y - y0) ** 2)
                        lon.append(x)
                        lat.append(y)
                        ele.append(z)
                        time.append(t)
                        name.append(os.path.basename(filepath))
                        dist.append(d)
                        x0, y0, d0 = x, y, d
            
            df = pd.DataFrame({
                "lon": lon, "lat": lat, "ele": ele,
                "time": time, "name": name, "dist": dist
            })
        
        elif filepath.endswith(".fit"):
            try:
                conv = fit2gpx.Converter()
                df_lap, df = conv.fit_to_dataframes(fname=filepath)
            except Exception as e:
                print(f"Error in FIT file ({filepath}): {e}")
                continue
            
            df["name"] = os.path.basename(filepath)
            dist = []
            
            for i in range(len(df)):
                if i < 1:
                    x0, y0, d0 = df["longitude"].iloc[0], df["latitude"].iloc[0], 0
                x, y = df["longitude"].iloc[i], df["latitude"].iloc[i]
                d = d0 + math.sqrt((x - x0) ** 2 + (y - y0) ** 2)
                dist.append(d)
                x0, y0, d0 = x, y, d
            
            df["dist"] = dist
            df = df.rename(columns={
                "longitude": "lon", "latitude": "lat",
                "altitude": "ele", "timestamp": "time"
            })
            df = df[["lon", "lat", "ele", "time", "name", "dist"]]
        
        else:
            continue
        
        dfs.append(df)
    
    # Process time columns
    for i in range(len(dfs)):
        if "time" in dfs[i].columns:
            dfs[i]["time"] = pd.to_datetime(dfs[i]["time"].astype(str), utc=True)
    
    dfs = [df for df in dfs if not df.empty]
    
    if not dfs:
        print("No data to process")
        return None
    
    df_all = pd.concat(dfs, ignore_index=True)
    df_all["time"] = pd.to_datetime(df_all["time"], utc=True)
    df_all["date"] = df_all["time"].dt.date
    df_all["hour"] = df_all["time"].dt.hour
    
    print(f"Successfully processed {len(df_all)} data points from {df_all['name'].nunique()} activities")
    
    # Create visualization if requested
    if plot:
        sns.set(style="whitegrid")
        
        start_times = (
            df_all.groupby("name").agg({"time": "min"}).reset_index().sort_values("time")
        )
        ncol = math.ceil(math.sqrt(len(start_times)))
        
        g = sns.FacetGrid(
            data=df_all,
            col="name",
            col_wrap=ncol,
            col_order=start_times["name"],
            sharex=False,
            sharey=False,
            height=2.5
        )
        
        g.map(plt.plot, "lon", "lat", color="black", linewidth=1)
        
        g.set(xlabel=None, ylabel=None, xticks=[], yticks=[], xticklabels=[], yticklabels=[])
        g.set_titles(col_template="", row_template="")
        sns.despine(left=True, bottom=True)
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
        plt.show()
    
    # Return data if requested
    if return_data:
        return df_all
    else:
        return None