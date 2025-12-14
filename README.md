# Education-Data-Simulation-and-Analysis 

This repository houses a data science demonstration project using simulated data in the K–12 private education sector. 

## Data setup 
Download required Census / TIGER shapefiles:

```bash
python -m src.ingestion.download_all_census
```

Generate city-zip mappings via spatial join:
```bash
python -m src.features.build_city_zip_map
```

Run simulation:
```bash
python -m src.simulation.run_simulation
```

## Repository structure

```text
.
├── README.md
├── .gitignore
├── requirements.txt
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── download_census.py
│   │   └── download_all_census.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_city_zip_map.py
│   └── simulation/
│       ├── __init__.py
│       ├── finaid_sim.py
│       └── run_simulation.py
└── data/
    ├── raw/
    │   ├── census/
    │   │   └── tiger/
    │   │       └── 2023/
    │   │           ├── place/
    │   │           │   └── README.md
    │   │           └── zcta520/
    │   │               └── README.md
    │   └── acs/
    └── processed/
        ├── geo/
        └── sim/
