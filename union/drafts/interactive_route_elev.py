"""
A4.py

This module provides geospatial functions to decode Strava polylines,
map routes using Python geospatial tools and generate elevation profiles.

Functions:
    - decode_polyline: Decode a Strava-encoded polyline into coordinates.
    - plot_route_map: Visualize a decoded route on an interactive map.
    - plot_elevation_profile: Generate a cross-section elevation profile.
    - interactive_dashboard: Creates an interactive dashboard containing the map and the elevation profile.

Requires:
    - polyline
    - folium
    - matplotlib
    - pandas
    - geopy
    - geodesic
    - requests
    - dash
    - plotly
"""

import polyline
import folium
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import pandas as pd

import requests
from geopy.distance import geodesic

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go


def decode_polyline(polyline_str):
    """
    Decode a polyline string into a list of (latitude, longitude) tuples.

    Args:
        polyline_str (str): Encoded polyline from Strava.

    Returns:
        list: List of (lat, lon) coordinate tuples.
    """
    return polyline.decode(polyline_str)


def plot_route_map(coords):
    """
    Plot a polyline route on a folium interactive map.

    Args:
        coords (list): List of (lat, lon) tuples.

    Returns:
        folium.Map: A folium map object with the route and markers.
    """
    # Get center point for initializing map
    center_lat, center_lon = coords[len(coords) // 2]
    start = coords[0]
    end = coords[-1]

    # Create a folium map centered on the polyline
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9.5)

    # Add the polyline to the map
    folium.PolyLine(locations=coords, color="blue", weight=5).add_to(m)
    folium.Marker(start, popup='Start', icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(end, popup='End', icon=folium.Icon(color='red')).add_to(m)

    return m


def plot_elevation_profile(coords):
    """
    Generate and display an elevation profile for a set of coordinates.

    Args:
        coords (list of tuple): List of (latitude, longitude) pairs.

    Returns:
        None: Displays a matplotlib elevation profile.
    """
    url = "https://api.open-elevation.com/api/v1/lookup"
    locations = [{"latitude": lat, "longitude": lon} for lat, lon in coords]
    response = requests.post(url, json={"locations": locations})

    if not response.ok:
        print("Error fetching elevation data.")
        return

    # Extract elevation data
    elevation_data = [result["elevation"] for result in response.json()["results"]]

    # Compute cumulative distance
    distances = [0]
    for i in range(1, len(coords)):
        d = geodesic(coords[i - 1], coords[i]).meters / 1000
        distances.append(distances[-1] + d)

    # Plot elevation profile
    plt.figure(figsize=(12, 6))
    plt.plot(distances, elevation_data, color="green")
    plt.xlabel("Distance (km)")
    plt.ylabel("Elevation (m)")
    plt.title("Elevation Profile of the Route")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def interactive_dashboard(coords):
    """
    Creates an interactive dashboard
    
    Args:
        coords (list): List of (lat, lon) tuples.

    Returns:
        Dashboard with an interactive map and an interactive cross-section.
    
    """
    from dash import Dash, dcc, html
    from dash.dependencies import Input, Output
    import plotly.graph_objects as go
    import polyline
    import requests
    from geopy.distance import geodesic    

    # Get elevations from open-elevation
    locations = [{"latitude": lat, "longitude": lon} for lat, lon in coords]
    url = "https://api.open-elevation.com/api/v1/lookup"
    response = requests.post(url, json={"locations": locations})
    if response.ok:
        elevations = [r['elevation'] for r in response.json()['results']]
    else:
        elevations = [0]*len(coords)

    # Calculate distances
    distances = [0]
    for i in range(1, len(coords)):
        d = geodesic(coords[i - 1], coords[i]).meters
        distances.append(distances[-1] + d)

    app = Dash(__name__)

    app.layout = html.Div([
        html.H3("Route Map & Elevation Profile Interactive Dashboard"),
    
        html.Div([
            dcc.Graph(id="map-graph", figure={}, style={"width": "100%", "height": "400px"}),
            dcc.Graph(id="elevation-graph", figure={}, style={"width": "100%", "height": "300px"}),
        ], style={"maxWidth": "900px", "margin": "auto"})  # constrain layout width and center it
    ])
    def create_map_figure(selected_index=None):
        latitudes = [c[0] for c in coords]
        longitudes = [c[1] for c in coords]
        marker_colors = ['red' if i == selected_index else 'blue' for i in range(len(coords))]
    
        # Main polyline + all route points
        fig = go.Figure(go.Scattermapbox(
            mode="lines+markers",
            lat=latitudes,
            lon=longitudes,
            marker=dict(size=8, color=marker_colors),
            line=dict(width=4, color="blue"),
            hoverinfo="text",
            text=[f"Point {i}<br>Lat: {lat:.5f}<br>Lon: {lon:.5f}<br>Elev: {elev} m"
                  for i, (lat, lon, elev) in enumerate(zip(latitudes, longitudes, elevations))],
            showlegend=False
        ))
    
        # Add Start marker (larger and with icon-like color)
        fig.add_trace(go.Scattermapbox(
            mode="markers+text",
            lat=[latitudes[0]],
            lon=[longitudes[0]],
            marker=dict(size=16, color='green'),
            text=["Start"],
            textfont=dict(size=14, color='black'),
            textposition="bottom right",
            hoverinfo="text",
            name="Start",
            showlegend=False
        ))
    
        # Add End marker
        fig.add_trace(go.Scattermapbox(
            mode="markers+text",
            lat=[latitudes[-1]],
            lon=[longitudes[-1]],
            marker=dict(size=16, color='red'),
            text=["End"],
            textfont=dict(size=14, color='black'),
            textposition="bottom right",
            hoverinfo="text",
            name="End",
            showlegend=False
        ))
    
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_zoom=8,
            mapbox_center={"lat": latitudes[len(latitudes)//2], "lon": longitudes[len(longitudes)//2]},
            margin={"r":0,"t":0,"l":0,"b":0},
            hovermode='closest'
        )
    
        return fig



    def create_elevation_figure(selected_index=None):
        marker_colors = ['red' if i == selected_index else 'green' for i in range(len(coords))]
        fig = go.Figure(go.Scatter(
            x=distances,
            y=elevations,
            mode="lines+markers",
            marker=dict(color=marker_colors),
            line=dict(color="green"),
            hoverinfo="text",
            text=[f"Distance: {dist:.1f} m<br>Elevation: {elev} m"
                  for dist, elev in zip(distances, elevations)]
        ))
        fig.update_layout(
            margin={"r":0,"t":30,"l":40,"b":40},
            xaxis_title="Distance (m)",
            yaxis_title="Elevation (m)",
            xaxis_tickformat=".2~s",
            hovermode='closest'
        )
        return fig

    @app.callback(
        Output("map-graph", "figure"),
        Output("elevation-graph", "figure"),
        Input("map-graph", "hoverData"),
        Input("elevation-graph", "hoverData")
    )
    def update_highlight(map_hover, elevation_hover):
        ctx = dash.callback_context
        if not ctx.triggered:
            idx = None
        else:
            triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if triggered_id == 'map-graph' and map_hover:
                idx = map_hover['points'][0]['pointIndex']
            elif triggered_id == 'elevation-graph' and elevation_hover:
                idx = elevation_hover['points'][0]['pointIndex']
            else:
                idx = None
        return create_map_figure(selected_index=idx), create_elevation_figure(selected_index=idx)

    app.run(debug=True)


