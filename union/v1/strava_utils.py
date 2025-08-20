# strava_utils.py
import pandas as pd
import numpy as np
import polyline
import folium
from folium.plugins import HeatMap
from folium import FeatureGroup, LayerControl
import plotly.express as px

# -----------------------
# Load Data
# -----------------------
def load_data(file_path='data/nicole_strava.csv'):
    df = pd.read_csv(file_path)
    df['distance_km'] = df['distance'] / 1000 
    df['moving_time'] = pd.to_timedelta(df['moving_time'], unit='s')
    df['start_date_local'] = pd.to_datetime(df['start_date_local'], errors='coerce')

    # Extract lat/lon safely
    def extract_latlon(x):
        if pd.isna(x):
            return pd.Series([np.nan, np.nan])
        try:
            return pd.Series(eval(x))
        except:
            return pd.Series([np.nan, np.nan])

    latlon_df = df['start_latlng'].apply(extract_latlon)
    latlon_df.columns = ['lat', 'lon']
    df = df.join(latlon_df)
    return df

# -----------------------
# Folium Map
# -----------------------
def create_folium_map(dff, output_file="map.html"):
    gps_points = []
    for _, row in dff.iterrows():
        poly = row.get('map.summary_polyline', None)
        if pd.notna(poly):
            coords = polyline.decode(poly)
            gps_points.extend(coords)

    if not gps_points:
        center_lat, center_lon = 0, 0
        m = folium.Map(location=[center_lat, center_lon], zoom_start=2, tiles='CartoDB positron')
    else:
        center_lat = np.mean([p[0] for p in gps_points])
        center_lon = np.mean([p[1] for p in gps_points])
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='CartoDB positron')
        min_lat, max_lat = min(p[0] for p in gps_points), max(p[0] for p in gps_points)
        min_lon, max_lon = min(p[1] for p in gps_points), max(p[1] for p in gps_points)
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    # Routes
    routes_layer = FeatureGroup(name='Routes', show=True)
    for _, row in dff.iterrows():
        poly = row.get('map.summary_polyline', None)
        if pd.notna(poly):
            coords = polyline.decode(poly)
            if coords:
                folium.PolyLine(coords, color='blue', weight=3).add_to(routes_layer)
    routes_layer.add_to(m)

    # Heatmap
    heatmap_layer = FeatureGroup(name='Heatmap', show=False)
    if gps_points:
        HeatMap(gps_points, radius=8, blur=7, min_opacity=0.4).add_to(heatmap_layer)
    heatmap_layer.add_to(m)

    LayerControl(collapsed=False).add_to(m)
    m.save(output_file)
    return output_file

# -----------------------
# Heatmap Figure
# -----------------------
def create_heatmap_figure(dff):
    dff['hour'] = dff['start_date_local'].dt.hour
    dff['day'] = dff['start_date_local'].dt.day_name()
    
    # Group by hour (rows) and day (columns)
    workout_df = dff.groupby(['hour','day']).size().unstack(fill_value=0).reindex(columns=[
        'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'
    ])
    
    fig = px.imshow(
        workout_df,
        labels=dict(x="Day", y="Hour", color="Sessions"),
        color_continuous_scale='Reds'
    )
    
    # Hide axis titles, invert y-axis, move x-axis to top
    fig.update_layout(
        xaxis=dict(title='', side='top'),
        yaxis=dict(title='', autorange='reversed'),
        margin=dict(l=20, r=20, t=20, b=20),
        template='plotly_white'
    )
    
    return fig



# -----------------------
# Weekly Distance
# -----------------------
def weekly_distance_figure(dff):
    weekly_df = dff.groupby(pd.Grouper(key='start_date_local', freq='W'))['distance_km'].sum().reset_index()
    fig = px.line(weekly_df, x='start_date_local', y='distance_km', markers=True, title="Weekly Distance", template='plotly_white')
    fig.update_layout(margin=dict(l=20,r=20,t=20,b=20))
    return fig

# -----------------------
# Personal Bests Table
# -----------------------
def personal_bests_table(dff):
    personal_bests_df = dff.sort_values('distance_km', ascending=False).head(10)
    personal_bests_df_display = personal_bests_df.rename(columns={
        'start_date_local':'Date',
        'distance_km':'Total Distance (km)',
        'moving_time':'Elapsed Time'
    })[['Date','name','Total Distance (km)','Elapsed Time']]
    columns=[{"name": i, "id": i} for i in personal_bests_df_display.columns]
    data=personal_bests_df_display.to_dict('records')
    return columns, data

# -----------------------
# Metrics
# -----------------------
def compute_metrics(dff):
    total_distance = round(dff['distance_km'].sum(), 1)
    avg_pace = (dff['moving_time'].sum() / dff['distance_km'].sum()) if dff['distance_km'].sum() > 0 else pd.Timedelta(seconds=0)
    avg_pace_str = f"{int(avg_pace.total_seconds() // 60)}:{int(avg_pace.total_seconds() % 60):02d} min/km"
    total_elev = f"{int(dff['total_elevation_gain'].sum())} m"
    activity_count = len(dff)
    avg_speed = round((dff['distance_km'].sum() / (dff['moving_time'].sum().total_seconds() / 3600)), 1) if dff['moving_time'].sum().total_seconds() > 0 else 0
    return total_distance, avg_pace_str, total_elev, activity_count, avg_speed

