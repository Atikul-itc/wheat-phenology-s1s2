# Winter wheat growth stages from Sentinel-1, Sentinel-2 and meteorology

Code for the paper *Complementary contributions of Sentinel-1 and Sentinel-2 to winter wheat growth stage retrieval in Germany* (Hoque, Fatema and Geli, under review).

The code retrieves the dates of seven winter wheat growth stages (seeding to harvest) for fields around German Meteorological Service (DWD) phenology stations, 2017 to 2021. It compares Sentinel-1, Sentinel-2 and daily meteorology as inputs to a bidirectional GRU, evaluated with leave-one-year-out cross-validation.

## Repository layout

| Folder | Script | What it does | Paper section |
|---|---|---|---|
| `extraction/` | `s1_extract_gee.py` | Sentinel-1 field statistics in Google Earth Engine (border-noise mask, γ⁰, linear means) | 2.5 |
| | `s1_build_daily.py` | Daily S1 series per field (VV, VH, CR) from the exports | 2.5 |
| | `s2_extract_gee.py` | Sentinel-2 Level-1C field means with cloud and shadow masking, 9 bands and 4 indices | 2.4 |
| | `s2_build_daily.py` | Daily S2 series per field from the exports | 2.4 |
| `preprocessing/` | `dwd_meteorology.py` | DWD download, ten-station IDW with lapse rate, growing degree days from 1 September | 2.6 |
| | `build_tensors.py` | Daily model tensors (426 days × 19 features, masks, soft labels) | 3.1, 3.2 |
| | `data_checks.py` | Station and field counts (Table 1, Table S2) and field shape metrics | 2.2, 2.3 |
| `phenology_pipeline/` | package | Reads the daily files and reference dates, builds the tensors | 3.1 |
| `model/` | `network.py`, `train.py` | Multi-source BiGRU and the experiment grid | 3.4, 3.5 |
| `baselines/` | package | Gradient boosting on seasonal summaries, double logistic fit | 3.3 |
| `analysis/` | `evaluate.py` | Accuracy, paired differences, station-clustered bootstrap, cloud analysis | 3.6, 3.7 |
| `figures/` | `make_figures.py` | Figures 5 to 8 | 4 |

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/Atikul-itc/wheat-phenology-s1s2.git
cd wheat-phenology-s1s2
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp config.example.yaml config.yaml                      # then edit the paths
```

For the Earth Engine scripts, register a Google Cloud project for Earth Engine and set `gee.project` in `config.yaml`. The first run opens a browser for authentication.

## Data

| Input | Source |
|---|---|
| Phenology reference dates | DWD Climate Data Center, annual reporters (`opendata.dwd.de/climate_environment/CDC/observations_germany/phenology`) |
| Daily station weather | DWD Climate Data Center, daily `kl` and `more_precip` (downloaded by `dwd_meteorology.py`) |
| Sentinel-1 GRD, Sentinel-2 L1C | Google Earth Engine (`COPERNICUS/S1_GRD_FLOAT`, `COPERNICUS/S2_HARMONIZED`) |
| Winter wheat fields | Germany-wide crop type maps (Gessner et al., 2025), available from the data providers |

Three inputs are prepared outside this repository:

- **Field GeoPackages** (`paths.fields`, one per harvest year): winter wheat fields of at least 2 ha after a 20 m inward buffer, the ten nearest within 5 km of each station, with the columns `XID` (field id), `Stations_i` (DWD station id) and `Rank` (1 = nearest).
- **Field links** (`paths.metadata`, `metadata_S2_<year>.csv`): the same `XID`, `Stations_i`, `Rank` table as CSV.
- **Phenology master file** (`paths.phenology_master`): DWD winter wheat records (columns `Stations_i`, `Referenzjahr`, `Phase_id`, `Eintrittsdatum`, `Eintrittsdatum_QB`) for stations with at least three years of data in 2017 to 2021. The quality flag and stage-order filters are applied by `phenology_pipeline`.

No data are stored in this repository.

## Run order

All scripts run from the repository root and take `--config config.yaml`.

```bash
# 1. Satellite data (any machine with internet access)
python -m extraction.s1_extract_gee --config config.yaml            # dry run: lists the export chunks
python -m extraction.s1_extract_gee --config config.yaml --submit   # start Drive exports
python -m extraction.s1_extract_gee --config config.yaml --status   # monitor
python -m extraction.s2_extract_gee --config config.yaml --submit
#    download the Drive folders, then:
python -m extraction.s1_build_daily --config config.yaml
python -m extraction.s2_build_daily --config config.yaml --exports <folder with S2_wheat*.csv>

# 2. Meteorology and tensors
python -m preprocessing.dwd_meteorology --config config.yaml
python -m preprocessing.build_tensors --config config.yaml
python -m preprocessing.data_checks --config config.yaml counts

# 3. Models (GPU recommended; resumes if interrupted)
python -m model.train --config config.yaml --smoke                  # quick test
python -m model.train --config config.yaml                          # full grid, 5 seeds x 5 test years

# 4. Tables and figures
python -m analysis.evaluate --config config.yaml
python -m figures.make_figures --config config.yaml
```

The full grid has 525 model runs. On one NVIDIA A100 it takes about [x] hours; the gradient boosting baseline adds about 30 to 60 minutes on CPU.

## Main settings

| Setting | Value | Where |
|---|---|---|
| Season window | 1 September (year before harvest) to 31 October, 426 days | `phenology_pipeline/config.py` |
| S1 mask | incidence angle 31° to 45°, VV > −30 dB, VH > −35 dB, valid share ≥ 0.9 | `extraction/s1_*.py` |
| S2 cloud mask | scene cloud cover < 70%, cloud score > 20, shadow projection 200 to 10,000 m | `extraction/s2_extract_gee.py` |
| Meteorology | ten nearest stations, IDW power 0.5, lapse rate −0.0065 °C m⁻¹, GDD base 0 °C | `preprocessing/dwd_meteorology.py` |
| Labels | Gaussian, σ = 5 days | `phenology_pipeline/config.py` |
| Model | BiGRU, 32 units per direction, Adam 10⁻³, batch 32, 15 epochs, 5 seeds | `model/train.py` |
| Statistics | station-clustered bootstrap, 2,000 replicates, ±1 day equivalence margin | `analysis/evaluate.py` |

## Citation

If you use this code, please cite the paper (reference to be added on publication).

## License

MIT, see `LICENSE`.
