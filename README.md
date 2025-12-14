# Education-Data-Simulation-and-Analysis 

This repository houses a data science demonstration project using simulated data in the K–12 private education sector. 

## Data setup Download required Census / TIGER shapefiles:

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
