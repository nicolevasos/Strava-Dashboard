import requests
import pandas as pd
import numpy as np
import json
import math
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import time



class StravaHeatmapGenerator:
    """
    Generates frequency-based heatmaps from Strava activities for web display.
    Uses 1m spatial resolution with dynamic grid based on map bounds.
    """
    
    def __init__(self, client_id: str, client_secret: str, refresh_token: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.access_token = None
        self.token_expires_at = None
        
    def refresh_access_token(self) -> bool:
        """Refresh the Strava access token using refresh token"""
        url = "https://www.strava.com/oauth/token"
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'refresh_token': self.refresh_token,
            'grant_type': 'refresh_token'
        }
        
        try:
            response = requests.post(url, data=data)
            response.raise_for_status()
            token_data = response.json()
            
            self.access_token = token_data['access_token']
            self.refresh_token = token_data['refresh_token']  # Update refresh token
            self.token_expires_at = datetime.now() + timedelta(seconds=token_data['expires_in'])
            
            print(f"Token refreshed, expires at: {self.token_expires_at}")
            return True
            
        except requests.RequestException as e:
            print(f"Failed to refresh token: {e}")
            return False
    
    def ensure_valid_token(self) -> bool:
        """Ensure we have a valid access token"""
        if not self.access_token or not self.token_expires_at:
            return self.refresh_access_token()
        
        if datetime.now() >= self.token_expires_at - timedelta(minutes=5):
            return self.refresh_access_token()
        
        return True
    
    def fetch_strava_activities(self, limit: int = 200) -> List[Dict]:
        """Fetch all available Strava activities"""
        if not self.ensure_valid_token():
            raise Exception("Could not obtain valid access token")
        
        url = "https://www.strava.com/api/v3/athlete/activities"
        headers = {'Authorization': f'Bearer {self.access_token}'}
        
        all_activities = []
        page = 1
        per_page = 200  # Max allowed by Strava
        
        while True:
            params = {
                'page': page,
                'per_page': min(per_page, limit - len(all_activities))
            }
            
            try:
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                activities = response.json()
                
                if not activities:  # No more activities
                    break
                
                all_activities.extend(activities)
                print(f"Fetched {len(activities)} activities (page {page})")
                
                if len(all_activities) >= limit or len(activities) < per_page:
                    break
                
                page += 1
                time.sleep(0.1)  # Rate limiting courtesy
                
            except requests.RequestException as e:
                print(f"Error fetching activities: {e}")
                break
        
        print(f"Total activities fetched: {len(all_activities)}")
        return all_activities[:limit]
    
    def fetch_activity_streams(self, activity_id: int) -> Optional[Dict]:
        """Fetch GPS streams for a specific activity"""
        if not self.ensure_valid_token():
            return None
        
        url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
        headers = {'Authorization': f'Bearer {self.access_token}'}
        params = {
            'keys': 'latlng,time',
            'key_by_type': 'true'
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            print(f"Error fetching streams for activity {activity_id}: {e}")
            return None
    
    def extract_all_gps_points(self, activities: List[Dict]) -> pd.DataFrame:
        """Extract GPS points from all activities"""
        all_points = []
        
        for i, activity in enumerate(activities):
            activity_id = activity['id']
            activity_type = activity.get('type', 'Unknown')
            
            print(f"Processing activity {i+1}/{len(activities)}: {activity_type} (ID: {activity_id})")
            
            streams = self.fetch_activity_streams(activity_id)
            if not streams or 'latlng' not in streams:
                continue
            
            latlng_data = streams['latlng']['data']
            
            for point in latlng_data:
                all_points.append({
                    'lat': point[0],
                    'lng': point[1],
                    'activity_id': activity_id,
                    'activity_type': activity_type
                })
            
            time.sleep(0.1)  # Rate limiting
        
        df = pd.DataFrame(all_points)
        print(f"Total GPS points extracted: {len(df)}")
        return df
    
    def create_grid_bounds(self, bounds: Dict[str, float], resolution_m: float = 1.0) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Create grid bounds for given map bounds
        bounds: {'north': lat, 'south': lat, 'east': lng, 'west': lng}
        resolution_m: grid resolution in meters
        """
        # Approximate conversion: 1 degree ≈ 111,000 meters at equator
        # This is rough but sufficient for grid generation
        lat_center = (bounds['north'] + bounds['south']) / 2
        
        # Adjust for latitude (longitude degrees get smaller toward poles)
        meters_per_degree_lat = 111000
        meters_per_degree_lng = 111000 * math.cos(math.radians(lat_center))
        
        # Calculate grid resolution in degrees
        lat_resolution = resolution_m / meters_per_degree_lat
        lng_resolution = resolution_m / meters_per_degree_lng
        
        # Create grid arrays
        lat_grid = np.arange(bounds['south'], bounds['north'] + lat_resolution, lat_resolution)
        lng_grid = np.arange(bounds['west'], bounds['east'] + lng_resolution, lng_resolution)
        
        grid_info = {
            'lat_resolution': lat_resolution,
            'lng_resolution': lng_resolution,
            'lat_min': bounds['south'],
            'lng_min': bounds['west'],
            'lat_bins': len(lat_grid),
            'lng_bins': len(lng_grid)
        }
        
        print(f"Grid created: {len(lat_grid)} x {len(lng_grid)} cells ({len(lat_grid) * len(lng_grid)} total)")
        return lat_grid, lng_grid, grid_info
    
    def calculate_frequency_grid(self, gps_points: pd.DataFrame, bounds: Dict[str, float]) -> Dict:
        """Calculate frequency of GPS points in grid cells"""
        
        # Filter points to bounds
        mask = (
            (gps_points['lat'] >= bounds['south']) & 
            (gps_points['lat'] <= bounds['north']) &
            (gps_points['lng'] >= bounds['west']) & 
            (gps_points['lng'] <= bounds['east'])
        )
        filtered_points = gps_points[mask].copy()
        
        if len(filtered_points) == 0:
            print("No GPS points found in specified bounds")
            return {'heatmap_data': [], 'stats': {'total_points': 0, 'grid_cells': 0}}
        
        print(f"Points in bounds: {len(filtered_points)}")
        
        # Create grid
        lat_grid, lng_grid, grid_info = self.create_grid_bounds(bounds)
        
        # Assign points to grid cells using numpy digitize
        lat_indices = np.digitize(filtered_points['lat'], lat_grid) - 1
        lng_indices = np.digitize(filtered_points['lng'], lng_grid) - 1
        
        # Create frequency matrix
        frequency_matrix = np.zeros((len(lat_grid), len(lng_grid)))
        
        # Count frequencies
        for lat_idx, lng_idx in zip(lat_indices, lng_indices):
            if 0 <= lat_idx < len(lat_grid) and 0 <= lng_idx < len(lng_grid):
                frequency_matrix[lat_idx, lng_idx] += 1
        
        # Convert to heatmap data format for Leaflet
        heatmap_data = []
        max_frequency = frequency_matrix.max()
        
        if max_frequency > 0:
            for i in range(len(lat_grid)):
                for j in range(len(lng_grid)):
                    if frequency_matrix[i, j] > 0:
                        # Use grid cell center coordinates
                        lat = lat_grid[i]
                        lng = lng_grid[j]
                        # Normalize intensity to 0-1 scale
                        intensity = frequency_matrix[i, j] / max_frequency
                        heatmap_data.append([lat, lng, intensity])
        
        stats = {
            'total_points': len(filtered_points),
            'grid_cells': len(heatmap_data),
            'max_frequency': int(max_frequency),
            'bounds': bounds
        }
        
        print(f"Heatmap generated: {len(heatmap_data)} cells with data")
        return {'heatmap_data': heatmap_data, 'stats': stats}
    
    def generate_heatmap_for_bounds(self, bounds: Dict[str, float], activity_limit: int = 200) -> Dict:
        """
        Main method to generate heatmap data for given bounds
        bounds: {'north': lat, 'south': lat, 'east': lng, 'west': lng}
        """
        print("=== Strava Heatmap Generation ===")
        print(f"Bounds: {bounds}")
        
        # Fetch activities
        activities = self.fetch_strava_activities(limit=activity_limit)
        if not activities:
            return {'error': 'No activities found'}
        
        # Extract GPS points
        gps_points = self.extract_all_gps_points(activities)
        if gps_points.empty:
            return {'error': 'No GPS data found'}
        
        # Generate frequency grid
        result = self.calculate_frequency_grid(gps_points, bounds)
        
        print("=== Generation Complete ===")
        return result

# Example usage and test functions
def example_usage():
    """Example of how to use the StravaHeatmapGenerator"""
    
    # Initialize with your Strava API credentials
    generator = StravaHeatmapGenerator(
        client_id='null',
        client_secret='null',
        refresh_token='null'
    )
    
    # Define map bounds (Salzburg + Gaisberg area)
    salzburg_gaisberg_bounds = {
        'north': 47.8300,
        'south': 47.7800,
        'east': 13.1200,
        'west': 13.0000
    }
    
    # Generate heatmap
    result = generator.generate_heatmap_for_bounds(salzburg_gaisberg_bounds, activity_limit=50)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    # Print stats
    stats = result['stats']
    print(f"\nHeatmap Stats:")
    print(f"Total GPS points: {stats['total_points']}")
    print(f"Grid cells with data: {stats['grid_cells']}")
    print(f"Max frequency per cell: {stats['max_frequency']}")
    
    # Save to file for web use
    with open('heatmap_data.json', 'w') as f:
        json.dump(result['heatmap_data'], f)
    
    print(f"Heatmap data saved to heatmap_data.json")
    return result

if __name__ == "__main__":
    # Run example
    example_usage()

    # Debug token refresh
def test_token_refresh():
    generator = StravaHeatmapGenerator(
        client_id='null',
        client_secret='null',
        refresh_token='null'
    )
    
    success = generator.refresh_access_token()
    print(f"Token refresh successful: {success}")
    print(f"Access token: {generator.access_token[:20]}..." if generator.access_token else "No token")

# Run the test
test_token_refresh()
