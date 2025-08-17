"""
strava_analysis.py

This script processes and analyzes geospatial data collected from Strava activities.
It performs the following:
- Loads and cleans raw Strava data.
- Extracts relevant temporal and spatial features.
- Visualizes activities on an interactive folium map and plots using seaborn.

Requirements:
    * pandas
    * geopandas
    * folium
    * matplotlib
    * seaborn

This file can also be importaed as a module, and the fuctions which could be used are:
    * load_strava_data
    * plot_speed_distribution
    * map_activities
"""

import pandas as pd
import geopandas as gpd
import folium
import seaborn as sns
import matplotlib.pyplot as plt
from shapely.geometry import Point
import branca.colormap as cm



def load_strava_data(filepath):
    """
    Load and clean the Strava activity data from a CSV file.

    Args:
        filepath (str): Path to the Strava CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame with selected and formatted columns.
    """
    cols = [
        'name', 'upload_id', 'type', 'distance', 'moving_time',
        'average_speed', 'max_speed', 'total_elevation_gain',
        'start_date_local', 'start_latlng', 'end_latlng', 'map.summary_polyline'
    ]
    
    df = pd.read_csv(filepath)
    df = df[cols].copy()

    # Convert date-time and extract time separately
    df['start_date_local'] = pd.to_datetime(df['start_date_local'], errors='coerce')
    df['start_time'] = df['start_date_local'].dt.time
    df['start_date_local'] = df['start_date_local'].dt.date

    # Remove entries without lat/lng
    df = df[df['start_latlng'].notnull()]
    
    return df


def plot_speed_distribution(df):
    """
    Plot the distribution of average and maximum speed using seaborn.

    Args:
        df (pd.DataFrame): Cleaned Strava DataFrame.
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(df['average_speed'], bins=30, kde=True, label='Average Speed', color='blue')
    sns.histplot(df['max_speed'], bins=30, kde=True, label='Max Speed', color='red', alpha=0.6)
    plt.title('Distribution of Speeds in Activities')
    plt.xlabel('Speed (m/s)')
    plt.legend()
    plt.grid(True)
    plt.show()


def extract_lat_lon(latlng_str):
    """
    Safely extract latitude and longitude from a string in the form '[lat, lon]'.

    Args:
        latlng_str (str): A string representing a list with two floats (e.g., "[48.2, 16.3]").

    Returns:
        tuple: A tuple (lat, lon) as floats if valid, or (None, None) if parsing fails.
    """
    try:
        lat, lon = latlng_str.strip('[]').split(',')
        return float(lat), float(lon)
    except Exception:
        return None, None


def map_df(df, zoom_start=2):
    """
    Plot activity start and end points on an interactive folium map.

    Args:
        df (pd.DataFrame): DataFrame with 'start_latlng' and 'end_latlng' as stringified lists.
        zoom_start (int): Initial zoom level for the folium map.

    Returns:
        folium.Map: A folium map with start (blue) and end (red) points.
    """
    # Create Point geometries
    start_geometry = [Point(xy) for xy in zip(df['start_lon'], df['start_lat'])]
    end_geometry = [Point(xy) for xy in zip(df['end_lon'], df['end_lat'])]

    # Create GeoDataFrame with start points
    gdf = gpd.GeoDataFrame(df.copy(), geometry=start_geometry, crs="EPSG:4326")

    #Delete nan
    gdf = gdf.dropna(subset=['start_lat', 'start_lon','end_lat', 'end_lon'])

    # Create the folium map centered on mean start coordinates
    map_center = [0, 0]
    fmap = folium.Map(location=map_center, zoom_start=zoom_start)

    # Add start points (blue)
    for _, row in gdf.iterrows():
        folium.CircleMarker(
            location=[row['start_lat'], row['start_lon']],
            radius=5,
            popup=f"{row['name']} ({row['type']}) - Start",
            color='blue',
            fill=True,
            fill_opacity=0.7
        ).add_to(fmap)

        folium.CircleMarker(
            location=[row['end_lat'], row['end_lon']],
            radius=4,
            color='red',
            fill=True,
            fill_opacity=0.6,
            popup='End'
        ).add_to(fmap)

    return fmap

def map_activities_colored_by_speed(df, zoom_start=2):
    """
    Create a folium map with activity start points color-coded by average speed.

    Args:
        df (pd.DataFrame): DataFrame with 'start_latlng' and 'average_speed'.
        zoom_start (int): Initial zoom level of the folium map.

    Returns:
        folium.Map: A folium map with color-coded markers by average speed.
    """
    
    # Create geometry and GeoDataFrame
    geometry = [Point(xy) for xy in zip(df['start_lon'], df['start_lat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    #Delete nan
    gdf = gdf.dropna(subset=['start_lat', 'start_lon','end_lat', 'end_lon'])

    # Create base map centered on 0,0
    map_center = [0, 0]
    fmap = folium.Map(location=map_center, zoom_start=zoom_start)

    # Define color scale based on average speed
    speed_colormap = cm.linear.YlGnBu_09.scale(
        gdf['average_speed'].min(),
        gdf['average_speed'].max()
    )

    # Add colored markers
    for _, row in gdf.iterrows():
        folium.CircleMarker(
            location=[row['start_lat'], row['start_lon']],
            radius=5,
            color=speed_colormap(row['average_speed']),
            fill=True,
            fill_opacity=0.7,
            popup=f"{row['name']} ({row['average_speed']:.2f} m/s)"
        ).add_to(fmap)

    # Add legend to map
    speed_colormap.caption = 'Average Speed (m/s)'
    speed_colormap.add_to(fmap)

    return fmap

#######################################################################################
# Arunima cont
#######################################################################################

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster
from shapely.geometry import Point, Polygon
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from esda.getisord import G_Local
from esda.moran import Moran_Local
from libpysal.weights import KNN, DistanceBand
import contextily as ctx
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_strava_data(csv_path):
    """load and preprocess the csv"""

    df = pd.read_csv(csv_path)
    print(f"loaded {len(df)} of your raw activities")

    # parse coords + clean + create gdf
    coords = df['start_latlng'].str.extract(r'\[([^,]+),\s*([^\]]+)\]')
    df['start_lat'] = pd.to_numeric(coords[0], errors='coerce')
    df['start_lng'] = pd.to_numeric(coords[1], errors='coerce')

    df = df.dropna(subset=['start_lat', 'start_lng', 'distance', 'average_speed'])
    df = df[df['distance'] > 0]

    geometry = [Point(lng, lat) for lng, lat in zip(df['start_lng'], df['start_lat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

    # feature eng.
    gdf['distance_km'] = gdf['distance'] / 1000
    gdf['speed_kmh'] = gdf['average_speed'] * 3.6
    gdf['duration_hours'] = gdf['moving_time'] / 3600
    gdf['pace_min_km'] = 60 / gdf['speed_kmh']
    gdf['elevation_rate'] = gdf['total_elevation_gain'] / gdf['distance_km']

    # performance scores
    scaler = StandardScaler()
    performance_features = ['speed_kmh', 'distance_km', 'total_elevation_gain']
    available_features = [f for f in performance_features if f in gdf.columns and gdf[f].notna().any()]

    if available_features:
        scaled_features = scaler.fit_transform(gdf[available_features].fillna(0))
        gdf['performance_score'] = np.mean(scaled_features, axis=1)
        gdf['intensity_score'] = (
            gdf['speed_kmh'].rank(pct=True) * 0.4 +
            gdf['distance_km'].rank(pct=True) * 0.3 +
            gdf['total_elevation_gain'].rank(pct=True) * 0.3
        )

    # temporal features
    if 'start_date' in gdf.columns:
        gdf['datetime'] = pd.to_datetime(gdf['start_date'])
        gdf['hour'] = gdf['datetime'].dt.hour
        gdf['day_of_week'] = gdf['datetime'].dt.dayofweek
        gdf['month'] = gdf['datetime'].dt.month
        gdf['is_weekend'] = gdf['day_of_week'].isin([5, 6])
        gdf['season'] = gdf['month'].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3})

    # ctaegorize
    gdf['distance_category'] = pd.cut(gdf['distance_km'],
                                     bins=[0, 5, 15, 30, np.inf],
                                     labels=['Short', 'Medium', 'Long', 'Ultra'])

    gdf['speed_category'] = pd.cut(gdf['speed_kmh'],
                                  bins=[0, 10, 20, 30, np.inf],
                                  labels=['Slow', 'Moderate', 'Fast', 'Very Fast'])

    print(f"processed {len(gdf)} of your activities with enhanced features")
    return gdf

def advanced_hotspot_analysis(gdf, attributes=['intensity_score'], k_neighbors=8):
    """Multi-attribute hotspot analysis with Getis-Ord G* + Moran's I
    check https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-statistics/h-how-spatial-autocorrelation-moran-s-i-spatial-st.htm

    """

    results = {}

    for attr in attributes:
        if attr not in gdf.columns or gdf[attr].isna().all():
            continue

        clean_gdf = gdf.dropna(subset=[attr]).copy()

        if len(clean_gdf) < k_neighbors:
            continue

        try:
            # spatial weights
            w = KNN.from_dataframe(clean_gdf, k=k_neighbors)
            w.transform = 'R'

            y = clean_gdf[attr].values

            # getis hotspots
            g_star = G_Local(y, w, star=True)
            clean_gdf[f'g_star_{attr}'] = g_star.Gs
            clean_gdf[f'g_star_pval_{attr}'] = g_star.p_sim

            # moran's // spatial outliers
            moran_local = Moran_Local(y, w)
            clean_gdf[f'moran_i_{attr}'] = moran_local.Is
            clean_gdf[f'moran_pval_{attr}'] = moran_local.p_sim

            # combined classification
            clean_gdf[f'spatial_type_{attr}'] = 'not_significant'

            # hotspots + coldspots (G*)
            sig_hot = (clean_gdf[f'g_star_{attr}'] > 0) & (clean_gdf[f'g_star_pval_{attr}'] < 0.05)
            sig_cold = (clean_gdf[f'g_star_{attr}'] < 0) & (clean_gdf[f'g_star_pval_{attr}'] < 0.05)

            clean_gdf.loc[sig_hot, f'spatial_type_{attr}'] = 'hotspot'
            clean_gdf.loc[sig_cold, f'spatial_type_{attr}'] = 'coldspot'

            # outliers (moran)
            high_high = (moran_local.q == 1) & (clean_gdf[f'moran_pval_{attr}'] < 0.05)
            low_low = (moran_local.q == 3) & (clean_gdf[f'moran_pval_{attr}'] < 0.05)
            high_low = (moran_local.q == 2) & (clean_gdf[f'moran_pval_{attr}'] < 0.05)
            low_high = (moran_local.q == 4) & (clean_gdf[f'moran_pval_{attr}'] < 0.05)

            clean_gdf.loc[high_high, f'spatial_type_{attr}'] = 'high_high'
            clean_gdf.loc[low_low, f'spatial_type_{attr}'] = 'low_low'
            clean_gdf.loc[high_low, f'spatial_type_{attr}'] = 'high_low_outlier'
            clean_gdf.loc[low_high, f'spatial_type_{attr}'] = 'low_high_outlier'

            results[attr] = clean_gdf

            # summary
            spatial_counts = clean_gdf[f'spatial_type_{attr}'].value_counts()
            print(f"     hotspots: {spatial_counts.get('hotspot', 0)}")
            print(f"     coldspots: {spatial_counts.get('coldspot', 0)}")
            print(f"     spatial clusters: {spatial_counts.get('high_high', 0) + spatial_counts.get('low_low', 0)}")
            print(f"     spatial outliers: {spatial_counts.get('high_low_outlier', 0) + spatial_counts.get('low_high_outlier', 0)}")

        except Exception as e:
            print(f"     failed for {attr}: {e}")

    return results

def spatial_clustering_analysis(gdf, eps_km=1.5, min_samples=3):
    """DBSCAN clustering w cluster characterization"""

    gdf_utm = gdf.to_crs('EPSG:3857')
    coords = np.column_stack([gdf_utm.geometry.x, gdf_utm.geometry.y])

    # DBSCAN clustering
    eps_meters = eps_km * 1000
    clustering = DBSCAN(eps=eps_meters, min_samples=min_samples)
    cluster_labels = clustering.fit_predict(coords)

    gdf['cluster'] = cluster_labels

    # stats
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    print(f"found {n_clusters} clusters with {n_noise} noise points")

    # now characterize each cluster
    cluster_stats = []
    for cluster_id in set(cluster_labels):
        if cluster_id == -1:
            continue

        cluster_data = gdf[gdf['cluster'] == cluster_id]

        stats_dict = {
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'avg_distance_km': cluster_data['distance_km'].mean(),
            'avg_speed_kmh': cluster_data['speed_kmh'].mean(),
            'avg_elevation': cluster_data['total_elevation_gain'].mean(),
            'dominant_activity': cluster_data['sport_type'].mode().iloc[0] if 'sport_type' in cluster_data.columns else 'Unknown'
        }
        cluster_stats.append(stats_dict)

    cluster_df = pd.DataFrame(cluster_stats)

    return gdf, cluster_df

# VISUALIZATION FUNCTIONS

def create_activity_locations_plot(gdf, save_path='activity_locations.png'):
    """Create activity locations plot with intensity on basemap"""

    gdf_proj = gdf.to_crs('EPSG:3857')

    fig, ax = plt.subplots(figsize=(15, 10))

    gdf_proj.plot(ax=ax, alpha=0.7, markersize=25, c=gdf['intensity_score'],
                  cmap='viridis', legend=True, edgecolors='white', linewidth=0.5)

    # dont forget to add basemap!!!
    try:
        ctx.add_basemap(ax, crs=gdf_proj.crs, source=ctx.providers.CartoDB.Positron, alpha=0.8)
    except:
        try:
            ctx.add_basemap(ax, crs=gdf_proj.crs, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.8)
        except:
            pass  # if it fails

    ax.set_title('Activity locations (colored by intensity)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

def create_hotspots_plot(hotspot_results, save_path='spatial_hotspots.png'):
    """Create hotspots analysis plot with basemap"""

    if not hotspot_results or 'intensity_score' not in hotspot_results:
        print("   no hotspot data available")
        return
    hotspot_data = hotspot_results['intensity_score']

    hotspot_proj = hotspot_data.to_crs('EPSG:3857')
    fig, ax = plt.subplots(figsize=(15, 10))

    colors = {
        'hotspot': '#ff4444',
        'coldspot': '#4444ff',
        'high_high': '#ff8888',
        'low_low': '#8888ff',
        'high_low_outlier': '#ffaa44',
        'low_high_outlier': '#44ffaa',
        'not_significant': '#cccccc'
    }

    # plot points by spatial type
    for spatial_type, color in colors.items():
        subset = hotspot_proj[hotspot_proj['spatial_type_intensity_score'] == spatial_type]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, alpha=0.8, markersize=30,
                       label=spatial_type.replace('_', ' ').title(),
                       edgecolors='white', linewidth=0.8)

    # basemap
    try:
        ctx.add_basemap(ax, crs=hotspot_proj.crs, source=ctx.providers.CartoDB.Positron, alpha=0.8)
    except:
        try:
            ctx.add_basemap(ax, crs=hotspot_proj.crs, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.8)
        except:
            pass

    ax.set_title('Spatial Hotspots & Clusters Analysis', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


def create_performance_distributions_plot(gdf, save_path='performance_distributions.png'):
    """Create performance distribution plots"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Performance Distributions', fontsize=16, fontweight='bold')

    # distance distribution
    gdf['distance_km'].hist(bins=30, ax=axes[0,0], alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Distance distribution')
    axes[0,0].set_xlabel('Distance (km)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].grid(True, alpha=0.3)

    # speed distribution
    gdf['speed_kmh'].hist(bins=30, ax=axes[0,1], alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0,1].set_title('Speed distribution')
    axes[0,1].set_xlabel('Speed (km/h)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].grid(True, alpha=0.3)

    # elevation distribution
    gdf['total_elevation_gain'].hist(bins=30, ax=axes[1,0], alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1,0].set_title('Elevation gain distribution')
    axes[1,0].set_xlabel('Elevation gain (m)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].grid(True, alpha=0.3)

    # performance score distribution
    if 'performance_score' in gdf.columns:
        gdf['performance_score'].hist(bins=30, ax=axes[1,1], alpha=0.7, color='gold', edgecolor='black')
        axes[1,1].set_title('Performance score distribution')
        axes[1,1].set_xlabel('Performance score')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


def create_temporal_patterns_plot(gdf, save_path='temporal_patterns.png'):
    """Create temporal patterns plot"""
    if 'hour' not in gdf.columns:
        print("   no temporal data available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Temporal activity patterns', fontsize=16, fontweight='bold')

    # hourly patterns
    hourly_counts = gdf.groupby('hour').size()
    hourly_counts.plot(kind='bar', ax=axes[0,0], color='orange', alpha=0.7)
    axes[0,0].set_title('Activity by hour of day')
    axes[0,0].set_xlabel('Hour')
    axes[0,0].set_ylabel('Count')
    axes[0,0].tick_params(axis='x', rotation=0)
    axes[0,0].grid(True, alpha=0.3)

    # daily patterns
    if 'day_of_week' in gdf.columns:
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily_counts = gdf.groupby('day_of_week').size()
        daily_counts.index = [day_names[i] for i in daily_counts.index]
        daily_counts.plot(kind='bar', ax=axes[0,1], color='purple', alpha=0.7)
        axes[0,1].set_title('Activity by day of week')
        axes[0,1].set_xlabel('Day')
        axes[0,1].set_ylabel('Count')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)

    # monthly patterns
    if 'month' in gdf.columns:
        monthly_counts = gdf.groupby('month').size()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_counts.index = [month_names[i-1] for i in monthly_counts.index]
        monthly_counts.plot(kind='bar', ax=axes[1,0], color='green', alpha=0.7)
        axes[1,0].set_title('Activity by month')
        axes[1,0].set_xlabel('Month')
        axes[1,0].set_ylabel('Count')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3)

    # weekend vs weekday?
    if 'is_weekend' in gdf.columns:
        weekend_counts = gdf['is_weekend'].value_counts()
        weekend_counts.index = ['Weekday', 'Weekend']
        weekend_counts.plot(kind='pie', ax=axes[1,1], autopct='%1.1f%%', startangle=90)
        axes[1,1].set_title('Weekend vs weekday activities')
        axes[1,1].set_ylabel('')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


def create_correlation_heatmap(gdf, save_path='correlation_heatmap.png'):
    """Create correlation heatmap"""

    numeric_cols = ['distance_km', 'speed_kmh', 'total_elevation_gain', 'intensity_score', 'performance_score']
    available_cols = [col for col in numeric_cols if col in gdf.columns]

    if len(available_cols) < 2:
        print("   not enough numeric columns for correlation!")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    corr_matrix = gdf[available_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
               square=True, ax=ax, cbar_kws={'shrink': 0.8})

    ax.set_title('Performance metrics correlation matrix', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

def create_distance_speed_scatter(gdf, save_path='distance_speed_scatter.png'):
    """Create distance vs speed scatter plot"""

    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(gdf['distance_km'], gdf['speed_kmh'],
                        c=gdf['total_elevation_gain'], cmap='terrain',
                        alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('Distance (km)', fontsize=12)
    ax.set_ylabel('Speed (km/h)', fontsize=12)
    ax.set_title('Distance vs speed (colored by elevation gain)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Elevation gain (m)', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


def create_clusters_plot(gdf, save_path='spatial_clusters.png'):
    """Create spatial clusters plot with basemap"""
    if 'cluster' not in gdf.columns:
        print("   no cluster data available")
        return

    gdf_proj = gdf.to_crs('EPSG:3857')
    fig, ax = plt.subplots(figsize=(15, 10))

    unique_clusters = gdf['cluster'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))

    # plt
    for i, cluster in enumerate(unique_clusters):
        cluster_data = gdf_proj[gdf_proj['cluster'] == cluster]
        if cluster == -1:
            cluster_data.plot(ax=ax, color='lightgray', alpha=0.5, markersize=20,
                             label='Noise', edgecolors='black', linewidth=0.5)
        else:
            cluster_data.plot(ax=ax, color=colors[i], alpha=0.8, markersize=40,
                             label=f'Cluster {cluster}', edgecolors='white', linewidth=0.8)

    # base
    try:
        ctx.add_basemap(ax, crs=gdf_proj.crs, source=ctx.providers.CartoDB.Positron, alpha=0.8)
    except:
        try:
            ctx.add_basemap(ax, crs=gdf_proj.crs, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.8)
        except:
            pass

    ax.set_title('Spatial activity clusters', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


def create_interactive_map(gdf, hotspot_results=None, save_path='strava_interactive_map.html'):
    """Create comprehensive interactive map"""

    center_lat = gdf['start_lat'].median()
    center_lng = gdf['start_lng'].median()

    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=12,
        tiles='CartoDB Positron'
    )

    # layer 1: all activities with performance colors
    activity_group = folium.FeatureGroup(name='All activities', show=True)

    # Sample data to avoid overcrowding
    sample_size = min(500, len(gdf))
    sample_gdf = gdf.sample(n=sample_size) if len(gdf) > sample_size else gdf

    for _, row in sample_gdf.iterrows():
        # color by perf score
        if 'performance_score' in row and not pd.isna(row['performance_score']):
            if row['performance_score'] > 0.75:
                color = 'red'
                category = 'High performance'
            elif row['performance_score'] > 0.25:
                color = 'orange'
                category = 'Medium performance'
            else:
                color = 'blue'
                category = 'Low performance'
        else:
            color = 'gray'
            category = 'Unknown'

        folium.CircleMarker(
            location=[row['start_lat'], row['start_lng']],
            radius=6,
            color=color,
            fill=True,
            popup=f"""
            <div style="font-family: Arial; min-width: 200px;">
                <h4 style="margin: 0; color: {color};">{category}</h4>
                <hr style="margin: 5px 0;">
                <b>Sport:</b> {row.get('sport_type', 'Unknown')}<br>
                <b>Distance:</b> {row['distance_km']:.1f} km<br>
                <b>Speed:</b> {row['speed_kmh']:.1f} km/h<br>
                <b>Elevation:</b> {row['total_elevation_gain']:.0f} m<br>
                <b>Date:</b> {str(row.get('datetime', 'Unknown'))[:10]}
            </div>
            """,
            fillOpacity=0.7
        ).add_to(activity_group)

    activity_group.add_to(m)

    # layer 2: clusters if avl
    if 'cluster' in gdf.columns:
        cluster_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
                         'lightred', 'beige', 'darkblue', 'darkgreen']

        for cluster_id in gdf['cluster'].unique():
            if cluster_id == -1:
                continue

            cluster_data = gdf[gdf['cluster'] == cluster_id]
            color = cluster_colors[cluster_id % len(cluster_colors)]

            cluster_group = folium.FeatureGroup(name=f'Cluster {cluster_id}', show=False)

            for _, row in cluster_data.iterrows():
                folium.CircleMarker(
                    location=[row['start_lat'], row['start_lng']],
                    radius=8,
                    color=color,
                    fill=True,
                    popup=f"""
                    <div style="font-family: Arial; min-width: 200px;">
                        <h4 style="margin: 0; color: {color};">Cluster {cluster_id}</h4>
                        <hr style="margin: 5px 0;">
                        <b>Distance:</b> {row['distance_km']:.1f} km<br>
                        <b>Speed:</b> {row['speed_kmh']:.1f} km/h<br>
                        <b>Elevation:</b> {row['total_elevation_gain']:.0f} m
                    </div>
                    """,
                    fillOpacity=0.8
                ).add_to(cluster_group)

            cluster_group.add_to(m)

    # layer 3: hotspots if avl
    if hotspot_results and 'intensity_score' in hotspot_results:
        hotspot_data = hotspot_results['intensity_score']

        hotspot_colors = {
            'hotspot': 'red',
            'coldspot': 'blue',
            'high_high': 'darkred',
            'low_low': 'darkblue',
            'high_low_outlier': 'orange',
            'low_high_outlier': 'green'
        }

        hotspot_group = folium.FeatureGroup(name='Spatial patterns', show=False)

        for _, row in hotspot_data.iterrows():
            spatial_type = row['spatial_type_intensity_score']

            if spatial_type in hotspot_colors:
                folium.CircleMarker(
                    location=[row['start_lat'], row['start_lng']],
                    radius=10,
                    color=hotspot_colors[spatial_type],
                    fill=True,
                    popup=f"""
                    <div style="font-family: Arial; min-width: 200px;">
                        <h4 style="margin: 0; color: {hotspot_colors[spatial_type]};">{spatial_type.replace('_', ' ').title()}</h4>
                        <hr style="margin: 5px 0;">
                        <b>G* Statistic:</b> {row['g_star_intensity_score']:.3f}<br>
                        <b>P-value:</b> {row['g_star_pval_intensity_score']:.3f}<br>
                        <b>Distance:</b> {row['distance_km']:.1f} km<br>
                        <b>Speed:</b> {row['speed_kmh']:.1f} km/h
                    </div>
                    """,
                    fillOpacity=0.9
                ).add_to(hotspot_group)

        hotspot_group.add_to(m)

    folium.LayerControl().add_to(m)

    # title
    title_html = '''
                 <h3 align="center" style="font-size:20px; color: #2E8B57; font-family: Arial;">
                 <b>Strava Activity Analysis Dashboard</b>
                 </h3>
                 '''
    m.get_root().html.add_child(folium.Element(title_html))

    m.save(save_path)
    print(f"saved interactive map: {save_path}")

    return m

def create_dashboard_html(gdf, cluster_df=None, save_path='strava_dashboard.html'):
    """Create a beautiful haha HTML dashboard with all analysis results"""

    # summary statistics
    total_activities = len(gdf)
    total_distance = gdf['distance_km'].sum()
    avg_speed = gdf['speed_kmh'].mean()
    total_elevation = gdf['total_elevation_gain'].sum()

    # date range
    date_range = "Unknown"
    if 'datetime' in gdf.columns:
        start_date = gdf['datetime'].min().strftime('%Y-%m-%d')
        end_date = gdf['datetime'].max().strftime('%Y-%m-%d')
        date_range = f"{start_date} to {end_date}"

    # most common activity
    most_common_activity = "Unknown"
    if 'sport_type' in gdf.columns:
        most_common_activity = gdf['sport_type'].mode().iloc[0]

    # peak activity hour
    peak_hour = "Unknown"
    if 'hour' in gdf.columns:
        peak_hour = f"{gdf.groupby('hour').size().idxmax()}:00"

    # cluster info
    cluster_info = "No clusters found"
    if cluster_df is not None and len(cluster_df) > 0:
        n_clusters = len(cluster_df)
        largest_cluster = cluster_df.loc[cluster_df['size'].idxmax()]
        cluster_info = f"{n_clusters} clusters found. Largest: {largest_cluster['size']} activities ({largest_cluster['dominant_activity']})"

    # HTML dashboard
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Strava Activity Analysis Dashboard</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}

            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                backdrop-filter: blur(10px);
            }}

            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding-bottom: 20px;
                border-bottom: 3px solid #667eea;
            }}

            .header h1 {{
                color: #2c3e50;
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
            }}

            .header p {{
                color: #7f8c8d;
                font-size: 1.2em;
            }}

            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }}

            .stat-card {{
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
                transform: translateY(0);
                transition: all 0.3s ease;
            }}

            .stat-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 15px 30px rgba(0, 0, 0, 0.2);
            }}

            .stat-card:nth-child(2) {{
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            }}

            .stat-card:nth-child(3) {{
                background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                color: #2c3e50;
            }}

            .stat-card:nth-child(4) {{
                background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
                color: #2c3e50;
            }}

            .stat-card:nth-child(5) {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }}

            .stat-card:nth-child(6) {{
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            }}

            .stat-value {{
                font-size: 2.5em;
                font-weight: bold;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
            }}

            .stat-label {{
                font-size: 1.1em;
                opacity: 0.9;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}

            .charts-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 30px;
                margin-bottom: 40px;
            }}

            .chart-card {{
                background: white;
                border-radius: 15px;
                padding: 25px;
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.05);
                border: 1px solid #e1e8ed;
            }}

            .chart-title {{
                color: #2c3e50;
                font-size: 1.3em;
                font-weight: bold;
                margin-bottom: 20px;
                text-align: center;
                padding-bottom: 10px;
                border-bottom: 2px solid #ecf0f1;
            }}

            .chart-placeholder {{
                height: 300px;
                background: linear-gradient(45deg, #f8f9fa 25%, transparent 25%),
                            linear-gradient(-45deg, #f8f9fa 25%, transparent 25%),
                            linear-gradient(45deg, transparent 75%, #f8f9fa 75%),
                            linear-gradient(-45deg, transparent 75%, #f8f9fa 75%);
                background-size: 20px 20px;
                background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #7f8c8d;
                font-style: italic;
                border: 2px dashed #bdc3c7;
                flex-direction: column;
                text-align: center;
            }}

            .insights-section {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 15px;
                margin-bottom: 30px;
            }}

            .insights-title {{
                font-size: 1.8em;
                margin-bottom: 20px;
                text-align: center;
            }}

            .insights-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
            }}

            .insight-item {{
                background: rgba(255, 255, 255, 0.1);
                padding: 20px;
                border-radius: 10px;
                backdrop-filter: blur(5px);
            }}

            .insight-item h4 {{
                margin-bottom: 10px;
                font-size: 1.2em;
            }}

            .files-section {{
                background: #f8f9fa;
                padding: 25px;
                border-radius: 15px;
                border-left: 5px solid #667eea;
            }}

            .files-title {{
                color: #2c3e50;
                font-size: 1.5em;
                margin-bottom: 15px;
            }}

            .file-list {{
                list-style: none;
            }}

            .file-list li {{
                padding: 8px 0;
                border-bottom: 1px solid #dee2e6;
                color: #495057;
            }}

            .file-list li:last-child {{
                border-bottom: none;
            }}

            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 2px solid #ecf0f1;
                color: #7f8c8d;
            }}

            @media (max-width: 768px) {{
                .stats-grid {{
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                }}

                .charts-grid {{
                    grid-template-columns: 1fr;
                }}

                .stat-value {{
                    font-size: 2em;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Strava Activity Analysis Dashboard</h1>
                <p>Comprehensive analysis of your training data</p>
                <p><strong>Analysis period:</strong> {date_range}</p>
            </div>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{total_activities:,}</div>
                    <div class="stat-label">Total activities</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_distance:,.0f}</div>
                    <div class="stat-label">Total distance (km)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg_speed:.1f}</div>
                    <div class="stat-label">Avg speed (km/h)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_elevation:,.0f}</div>
                    <div class="stat-label">Total elevation (m)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{most_common_activity}</div>
                    <div class="stat-label">Top activity</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{peak_hour}</div>
                    <div class="stat-label">Peak hour</div>
                </div>
            </div>

            <div class="insights-section">
                <h3 class="insights-title">Key Insights</h3>
                <div class="insights-grid">
                    <div class="insight-item">
                        <h4>Activity Patterns</h4>
                        <p>Most active during {peak_hour} with {most_common_activity} being the dominant activity type.</p>
                    </div>
                    <div class="insight-item">
                        <h4>Performance</h4>
                        <p>Average speed of {avg_speed:.1f} km/h with total distance of {total_distance:,.0f} km covered.</p>
                    </div>
                    <div class="insight-item">
                        <h4>Elevation Challenge</h4>
                        <p>Conquered {total_elevation:,.0f} meters of elevation gain across all activities.</p>
                    </div>
                    <div class="insight-item">
                        <h4>Spatial analysis</h4>
                        <p>{cluster_info}</p>
                    </div>
                </div>
            </div>

            <div class="charts-grid">
                <div class="chart-card">
                    <h3 class="chart-title">Performance distributions</h3>
                    <div class="chart-placeholder">
                        Generated: performance_distributions.png<br>
                        <small>Distance, speed, elevation, and performance score distributions</small>
                    </div>
                </div>

                <div class="chart-card">
                    <h3 class="chart-title">Temporal patterns</h3>
                    <div class="chart-placeholder">
                        Generated: temporal_patterns.png<br>
                        <small>Activity patterns by hour, day, month, and season</small>
                    </div>
                </div>

                <div class="chart-card">
                    <h3 class="chart-title">Activity Locations</h3>
                    <div class="chart-placeholder">
                        Generated: activity_locations.png<br>
                        <small>Geographic distribution of all activities</small>
                    </div>
                </div>

                <div class="chart-card">
                    <h3 class="chart-title">Spatial hotspots</h3>
                    <div class="chart-placeholder">
                        Generated: spatial_hotspots.png<br>
                        <small>Statistical hotspots and spatial clustering analysis</small>
                    </div>
                </div>

                <div class="chart-card">
                    <h3 class="chart-title">Correlations</h3>
                    <div class="chart-placeholder">
                        Generated: correlation_heatmap.png<br>
                        <small>Performance metrics correlation matrix</small>
                    </div>
                </div>

                <div class="chart-card">
                    <h3 class="chart-title">Spatial clusters</h3>
                    <div class="chart-placeholder">
                        Generated: spatial_clusters.png<br>
                        <small>DBSCAN clustering of activity locations</small>
                    </div>
                </div>
            </div>

            <div class="files-section">
                <h3 class="files-title">Generated files</h3>
                <ul class="file-list">
                    <li><strong>performance_distributions.png</strong> - Distribution plots for key metrics</li>
                    <li><strong>temporal_patterns.png</strong> - Time-based activity analysis</li>
                    <li><strong>activity_locations.png</strong> - Geographic activity map</li>
                    <li><strong>spatial_hotspots.png</strong> - Statistical spatial analysis</li>
                    <li><strong>correlation_heatmap.png</strong> - Performance correlation matrix</li>
                    <li><strong>spatial_clusters.png</strong> - Activity location clusters</li>
                    <li><strong>strava_interactive_map.html</strong> - Interactive web map</li>
                    <li><strong>strava_comprehensive_report.txt</strong> - Detailed analysis report</li>
                    <li><strong>strava_enhanced_results.csv</strong> - Enhanced dataset with analysis</li>
                </ul>
            </div>

            <div class="footer">
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Strava Analysis Dashboard v2.0</p>
            </div>
        </div>
    </body>
    </html>
    """

    # Save dashboard
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"   saved dashboard: {save_path}")
    return save_path

def display_html_map_simple(html_file):
    """Simple method to display HTML map that works reliably"""
    import os
    from IPython.display import HTML, display

    if os.path.exists(html_file):
        print(f"opening {html_file}...")

        try:
            # read the HTML file
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # display directly
            display(HTML(html_content))
            print("   map displayed successfully!")

        except Exception as e:
            print(f"   display failed: {e}")

            # fallback
            abs_path = os.path.abspath(html_file)
            print(f"   open manually: {abs_path}")
            print(f"   or copy this to browser: file://{abs_path}")

    else:
        print(f"   file not found: {html_file}")

def generate_comprehensive_report(gdf, hotspot_results=None, cluster_df=None, correlations=None):
    """Generate detailed analysis report"""

    report = []
    report.append("=" * 60)
    report.append("COMPREHENSIVE STRAVA GEOSTATISTICAL ANALYSIS")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Basic statistics
    report.append("BASIC STATISTICS")
    report.append("-" * 30)
    report.append(f"Total activities: {len(gdf):,}")

    if 'datetime' in gdf.columns:
        date_range = gdf['datetime'].max() - gdf['datetime'].min()
        report.append(f"Analysis period: {date_range.days} days")
        report.append(f"Average activities per day: {len(gdf) / max(date_range.days, 1):.1f}")

    # Activity types
    if 'sport_type' in gdf.columns:
        report.append(f"\nActivity breakdown:")
        for sport, count in gdf['sport_type'].value_counts().head(5).items():
            pct = count / len(gdf) * 100
            report.append(f"  • {sport}: {count:,} ({pct:.1f}%)")

    # Performance metrics
    report.append(f"\nPERFORMANCE METRICS")
    report.append("-" * 30)
    report.append(f"Total distance: {gdf['distance_km'].sum():,.1f} km")
    report.append(f"Average distance: {gdf['distance_km'].mean():.1f} km")
    report.append(f"Average speed: {gdf['speed_kmh'].mean():.1f} km/h")
    report.append(f"Total elevation gain: {gdf['total_elevation_gain'].sum():,.0f} m")

    # Spatial analysis results
    if hotspot_results:
        report.append(f"\nSPATIAL HOTSPOT ANALYSIS")
        report.append("-" * 30)

        for attr, data in hotspot_results.items():
            spatial_counts = data[f'spatial_type_{attr}'].value_counts()
            report.append(f"\nAnalysis of {attr}:")
            report.append(f"  • Hotspots: {spatial_counts.get('hotspot', 0)}")
            report.append(f"  • Coldspots: {spatial_counts.get('coldspot', 0)}")
            report.append(f"  • High-high clusters: {spatial_counts.get('high_high', 0)}")
            report.append(f"  • Low-low clusters: {spatial_counts.get('low_low', 0)}")
            report.append(f"  • Spatial outliers: {spatial_counts.get('high_low_outlier', 0) + spatial_counts.get('low_high_outlier', 0)}")

    # Clustering results
    if cluster_df is not None and len(cluster_df) > 0:
        report.append(f"\nSPATIAL CLUSTERING")
        report.append("-" * 30)
        report.append(f"Number of clusters: {len(cluster_df)}")

        for _, cluster in cluster_df.head(5).iterrows():
            report.append(f"\nCluster {cluster['cluster_id']}:")
            report.append(f"  • Size: {cluster['size']} activities")
            report.append(f"  • Avg distance: {cluster['avg_distance_km']:.1f} km")
            report.append(f"  • Avg speed: {cluster['avg_speed_kmh']:.1f} km/h")
            report.append(f"  • Dominant activity: {cluster['dominant_activity']}")

    report.append("")
    report.append("=" * 60)

    # Save report
    with open('strava_comprehensive_report.txt', 'w') as f:
        f.write('\n'.join(report))

    print(f"   saved report: strava_comprehensive_report.txt")
    return '\n'.join(report)

def main_analysis(csv_path):
    """Main analysis pipeline with separated visualizations"""

    try:
        # 1. Load and preprocess data
        gdf = load_strava_data(csv_path)

        # 2. Advanced analyses
        hotspot_attributes = ['intensity_score', 'speed_kmh', 'distance_km']
        available_attributes = [attr for attr in hotspot_attributes if attr in gdf.columns]

        hotspot_results = advanced_hotspot_analysis(gdf, available_attributes)
        clustered_gdf, cluster_df = spatial_clustering_analysis(gdf)
        gdf = clustered_gdf

        # 3. Create individual visualizations
        create_activity_locations_plot(gdf)
        create_hotspots_plot(hotspot_results)
        create_performance_distributions_plot(gdf)
        create_temporal_patterns_plot(gdf)
        create_correlation_heatmap(gdf)
        create_distance_speed_scatter(gdf)
        create_clusters_plot(gdf)

        # 4. Create interactive map
        print("\nCreating interactive map")
        interactive_map = create_interactive_map(gdf, hotspot_results)

        # 5. Generate comprehensive report
        print("\nGenerating report...")
        report = generate_comprehensive_report(gdf, hotspot_results, cluster_df)

        # 6. Create dashboard
        print("\nCreating dashboard...")
        dashboard_path = create_dashboard_html(gdf, cluster_df)

        # 7. Export results
        print("\nExporting results as csv")
        gdf.to_csv('strava_enhanced_results.csv', index=False)

        if hotspot_results and available_attributes:
            primary_attr = available_attributes[0]
            significant_patterns = hotspot_results[primary_attr][
                hotspot_results[primary_attr][f'spatial_type_{primary_attr}'] != 'not_significant'
            ]
            if len(significant_patterns) > 0:
                significant_patterns.to_csv('strava_significant_patterns.csv', index=False)

        # 8. Display results
        print("\nAnalysis complete!")
        print("\ngenerated files:")
        print("• individual visualization PNGs (7 files)")
        print("• strava_interactive_map.html (interactive map)")
        print("• strava_dashboard.html (beautiful dashboard)")
        print("• strava_comprehensive_report.txt (detailed report)")
        print("• strava_enhanced_results.csv (enhanced dataset)")

        # interactive map
        display_html_map_simple('strava_interactive_map.html')
        print("   open this file in your browser !")

        return gdf, hotspot_results, cluster_df

    except Exception as e:
        print(f"analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None