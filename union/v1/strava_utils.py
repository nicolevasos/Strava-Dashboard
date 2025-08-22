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
    routes_layer = FeatureGroup(name='Routes', show=False)
    for _, row in dff.iterrows():
        poly = row.get('map.summary_polyline', None)
        if pd.notna(poly):
            coords = polyline.decode(poly)
            if coords:
                folium.PolyLine(coords, color='blue', weight=3).add_to(routes_layer)
    routes_layer.add_to(m)

    # Heatmap
    heatmap_layer = FeatureGroup(name='Heatmap', show=True)
    if gps_points:
        HeatMap(gps_points, radius=8, blur=7, min_opacity=0.4).add_to(heatmap_layer)
    heatmap_layer.add_to(m)

    LayerControl(collapsed=True).add_to(m)
    m.save(output_file)
    return output_file

# -----------------------
# Heatmap Figure
# -----------------------
def create_heatmap_figure(dff):
    dff['hour'] = dff['start_date_local'].dt.hour
    dff['day'] = dff['start_date_local'].dt.day_name()

    # Group by hour and day
    workout_df = dff.groupby(['hour','day']).size().unstack(fill_value=0).reindex(columns=[
        'Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'
    ])

    # Custom text: show count if > 0, else empty string
    custom_text = workout_df.where(workout_df > 0, "").astype(str).values

    # Hour labels
    hour_labels = {h: pd.to_datetime(str(h), format='%H').strftime('%I:%M %p') for h in range(24)}

    # Heatmap
    fig = px.imshow(
        workout_df,
        labels=dict(x="Day", y="Hour", color="Oranges"),
        color_continuous_scale='Reds'
    )

    # Inject custom text into the trace
    fig.update_traces(
        text=custom_text,
        texttemplate="%{text}",
        textfont=dict(color="#666666", size=10)
    )

    vmin, vmax = workout_df.values.min(), workout_df.values.max()

    fig.update_layout(
        font=dict(color="#666666"),
        xaxis=dict(title='', side='top', tickfont=dict(color="#666666")),
        yaxis=dict(
            title='',
            autorange='reversed',
            tickmode='array',
            tickvals=list(range(24)),
            ticktext=[hour_labels[h] for h in range(24)],
            tickfont=dict(color="#666666")
        ),
        margin=dict(l=40, r=20, t=40, b=60),
        template='plotly_white',
        coloraxis_colorbar=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.3,
            xanchor='left',   
            x=0,              
            tickvals=[vmin, vmax],
            ticktext=['low', 'high'],
            title="",
            ticks="outside",
            len=0.4,              
            thickness=10,         
            tickfont=dict(color="#666666", size=10) 
        )
    )

    

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig

# -----------------------
# Weekly Distance
# -----------------------
def weekly_distance_figure(dff):
    # Aggregate total distance per week
    weekly_df = dff.groupby(pd.Grouper(key='start_date_local', freq='W'))['distance_km'].sum().reset_index()
    weekly_df['distance_km'] = weekly_df['distance_km'].round(2)  # round distances

    # Create line chart with filled area
    fig = px.line(
        weekly_df,
        x='start_date_local',
        y='distance_km',
        markers=True,
        template='plotly_white',
        labels={
            'start_date_local': 'Week',
            'distance_km': 'Total Distance (km)'
        }
    )

    # Fill area below the curve
    fig.update_traces(fill='tozeroy', line_color='#FF6B6B', marker_color='#FF6B6B')

    # Improve layout
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(tickformat='%d/%m/%Y'),  # format dates as d/mm/yyyy
        yaxis=dict(tickformat=',.2f')       # two decimals on y-axis
    )

    return fig


def format_timedelta_hm(td):
    """
    Convert pandas Timedelta to H:MM, rounded down to minutes.
    """
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    return f"{hours}:{minutes:02d}"

def personal_bests_table(dff):
    personal_bests_df = dff.sort_values('distance_km', ascending=False).head(10)

    # Format date as d/mm/yyyy
    personal_bests_df['Date'] = personal_bests_df['start_date_local'].dt.strftime('%-d/%m/%Y')

    # Format elapsed time as H:MM
    personal_bests_df['Elapsed Time'] = personal_bests_df['moving_time'].apply(format_timedelta_hm)

    # Round distance to 2 decimals
    personal_bests_df['distance_km'] = personal_bests_df['distance_km'].round(2)

    personal_bests_df_display = personal_bests_df.rename(columns={
        'distance_km':'Total Distance (km)',
    })[['Date','name','Total Distance (km)','Elapsed Time']]

    columns = [{"name": i, "id": i} for i in personal_bests_df_display.columns]
    data = personal_bests_df_display.to_dict('records')
    
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

