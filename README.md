# Strava Dashboard

A personalised web-app dashboard that leverages the Strava API to extract, analyse, and visualise physical activity data. This interactive dashboard provides insights into workout patterns, performance metrics, and spatial activity distribution through heat maps and geospatial analysis.
This dashboard serves as both a motivational tool and a decision-making support system for fitness enthusiasts and athletes seeking data-driven training insights.

## View of the dashboard
<img width="1600" height="788" alt="image" src="https://github.com/user-attachments/assets/1e457026-ade0-461f-9b08-5fec1d0ca2f4" />


## Features

- Interactive activity filtering by type (hike, run, bike) and date ranges
- Activity heat maps for spatial visualisation of frequency and distribution
- Performance analytics tracking workout preferences and speed evolution
- Geospatial analysis with hotspot identification using Getis-Ord statistics

## Project Structure

```
Strava-Dashboard/
├── ss/                          # Superseded old components
├── union/                       # Main executable directory
│   ├── drafts/                  # Legacy dashboard elements and inputs
│   ├── strava_env.yaml          # Environment configuration
│   └── v1/                      # Current implementation (union v1)
│       ├── assets/              # Static files and resources
│       ├── data/                # Activity data and processed files
│       ├── main.ipynb           # Core dashboard notebook
│       ├── map.html             # Generated map visualisation
│       └── strava_utils.py      # Utility functions
├── proposal.md                  # Original project proposal
└── README.md
```

**Note:** The current main implementation resides in the `union v1` with `main.ipynb` as the primary executable.

## Installation

Clone the repository:
```bash
git clone https://github.com/nicolevasos/Strava-Dashboard.git
cd Strava-Dashboard
```

Create and activate the conda environment:
```bash
conda env create -f strava_env.yaml
conda activate strava_env
```

## Usage

**Important:** The Jupyter Notebook must be executed to generate the dashboard before visualisation.

1. Run the main notebook to process data and generate the dashboard
2. View the dashboard using one of the following methods:
   - **Notebook output:** View visualisations directly in the Jupyter Notebook canvas
   - **Local server:** Access the interactive dashboard at `http://127.0.0.1:8050/`

The dashboard allows users to filter activities by type and date range, explore spatial patterns through heat maps, and analyse performance trends over time.

## Technical Stack

- **Backend:** Python, Jupyter Notebook
- **API:** Strava API
- **Geospatial Analysis:** GeoPandas, Folium
- **Visualisation:** Matplotlib, Seaborn, Plotly, Dash
- **Statistical Analysis:** Getis-Ord statistics
- **Mapping:** LeafLet, MapBox, Carto
- **Environment:** Conda

## Contributors

| Name | GitHub | Role |
|------|--------|------|
| Nicole Salazar-Cuellar | [@nicolevasos](https://github.com/nicolevasos) | Project initialization, core package setup, dashboard layout design |
| Angelica Maria Moreno | [@angelicarjs](https://github.com/Angelicarjs) | Statistics and geo-statistics, workout visualization, performance analytics |
| Arunima Sen | [@arunima-sen](https://github.com/arunima-sen) | Hotspot analysis, Getis-Ord statistics, spatial insights through heatmaps |
| Emese Gojdar | [@placcky](https://github.com/placcky) | Data preparation for dashboard visualization |
| Maria Anna Fedyszyn | [@maria-anna-gis](https://github.com/maria-anna-gis) | Data preparation for heat maps and visualisation |
