# Hispaniola_Seismicity_ML
Unified machine-learning derived earthquake catalog for Hispaniola, combining regional and local datasets. This repository includes Python scripts, catalogs, and supplemental materials supporting the study “Remarkable crustal and slab seismicity in Hispaniola revealed through a unified machine-learning derived earthquake catalog.”

## Requirements
These codes rely on the easyQuake python package (https://github.com/jakewalter/easyQuake) which requires:

- Python 3.7
- Tensorflow-gpu 2.2
- Obspy
- Keras

### Download Data
To build the catalog, raw waveform data must first be downloaded from publicly available repositories of the International Federation of Digital Seismograph Networks (FDSN).
```
from easyQuake import download_mseed
from easyQuake import daterange
from datetime import date

maxkm = 300
maxdist=300
lat_a = 15
lat_b = 23
lon_a = -75.5
lon_b = -67

start_date = date(2018, 1, 1)
end_date = date(2019, 1, 1)

project_code = '2018'
project_folder = '/data/DRproject/2018'
for single_date in daterange(start_date, end_date):
    print(single_date.strftime("%Y-%m-%d"))
    dirname = single_date.strftime("%Y%m%d")
    download_mseed(dirname=dirname, project_folder=project_folder, single_date=single_date, minlat=lat_a, maxlat=lat_b, minlon=lon_a, maxlon=lon_b, raspberry_shake=True)
```
### Phase Detection
easyQuake integrates three deep-learning phase pickers (EQTransformer, PhaseNet, GPD). These can be selected in the detection_continuous function to perform event detection and seismic phase picking.
```
from easyQuake import detection_continuous
from easyQuake import daterange
from datetime import date
start_date = date(2018, 1, 1)
end_date = date(2019, 1, 1)

project_code = '2018'
project_folder = '/data/DRproject/2018'
for single_date in daterange(start_date, end_date):
    dirname = single_date.strftime("%Y%m%d")
    #GPD
    detection_continuous(dirname=dirname, project_folder=project_folder, project_code=project_code, single_date=single_date, machine=True,local=True)
    #EQTransformer
    #detection_continuous(dirname=dirname, project_folder=project_folder, project_code=project_code, machine=True, machine_picker='EQTransformer', local=True, single_date=single_date)
    #PhaseNet
    #detection_continuous(dirname=dirname, project_folder=project_folder, project_code=project_code, machine=True, machine_picker='PhaseNet', local=True, single_date=single_date, fullpath_python='/home/luis/anaconda3/envs/easyquake/bin/python')
```
### Event Association and Magnitude Calculation
This step uses the default associator within easyQuake and the Python multiprocessing package to parallelize the job.

```
from easyQuake import association_continuous
from easyQuake import daterange
from easyQuake import combine_associated
from datetime import date
from easyQuake import magnitude_quakeml
from easyQuake import simple_cat_df
import matplotlib.pyplot as plt
from multiprocessing import Pool

start_date = date(2018, 1, 1)
end_date = date(2019, 1, 1)
maxdist = 300
maxkm = 300
pool = Pool(40)

project_code = '2018'
project_folder = '/data/DRproject/2018'

for single_date in daterange(start_date, end_date):
    dirname = single_date.strftime("%Y%m%d")
    #GPD
    pool.apply_async(association_continuous, (dirname, project_folder, project_code, maxdist, maxkm, single_date, True, 3, 1, True, None, None, None, None, 'DRMODEL.tvel', True))
    #EarthQuake Transformer
    #pool.apply_async(association_continuous, (dirname, project_folder, project_code, maxdist, maxkm, single_date, True, 3, 1, True, 'EQTransformer', None, None, None, None, True))
    #PhaseNet
    #pool.apply_async(association_continuous, (dirname, project_folder, project_code, maxdist, maxkm, single_date, True, 3, 1, True, 'PhaseNet', None, None, None, 'DRMODEL.tvel', True))

pool.close()
pool.join()

cat, dfs = combine_associated(project_folder=project_folder, project_code=project_code, machine_picker='GPD')
#cat, dfs = combine_associated(project_folder=project_folder, project_code=project_code, machine_picker='EQTransformer')
#cat, dfs = combine_associated(project_folder=project_folder, project_code=project_code, machine_picker='PhaseNet')

# Estimate magnitudes and write the QuakeML catalog to disk
cat = magnitude_quakeml(cat=cat, project_folder=project_folder, plot_event=False)
cat.write('catalog-2018-GPD.xml',format='QUAKEML')

catdf = simple_cat_df(cat)
#plt.figure()
#plt.plot(catdf.index,catdf.magnitude,'.')

print(cat.__str__(print_all=True))
#print(catdf)
```
You can also use PyOcto associator (https://github.com/yetinam/pyocto) instead of the default easyQuake associator:
```
import pyocto
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# Read picks from the deep-learning phase picker
picks = pd.read_csv("/data/DRproject/2018/gpd-picks-2018-pyocto.csv")
print(picks)

# Convert the 'time' column to datetime format
picks["time"]=pd.to_datetime(picks["time"])
print(picks)

# Convert pick times from datetime to UNIX timestamps (floats in seconds)
picks["time"] = picks["time"].apply(lambda x: x.timestamp())
print(picks)

# Read station information
stations = pd.read_csv("/data/DRproject/2018/stations-list-2018-pyocto.csv")
print(stations)

# Create a 1D velocity model from input layers
layers = pd.read_csv("DRMODEL.csv")
layers
model_path = "DR_velocity_model"
pyocto.VelocityModel1D.create_model(layers, 1., 400, 250, model_path)

# Load the velocity model
velocity_model = pyocto.VelocityModel1D(model_path, tolerance=2.0)

# Create an associator instance and run phase association
associator = pyocto.OctoAssociator.from_area(
    lat=(15, 23),
    lon=(-75.5, -66.5),
    zlim=(0, 250),
    time_before=300,
    velocity_model=velocity_model,
    n_picks=6,
    n_p_and_s_picks=2,
)

print(associator.crs)
associator.transform_stations(stations)
events_1d, assignments_1d = associator.associate(picks, stations)

# Convert event times back to datetime (UTC) and transform coordinates to lat/lon
associator.transform_events(events_1d)
events_1d["time"] = events_1d["time"].apply(datetime.datetime.fromtimestamp, tz=datetime.timezone.utc)
print(events_1d)
print(events_1d.to_string())
#pd.set_option('display.max_columns', None)
#print(events_1d)
events_1d.to_csv('catalog-pyocto-gpd-2018.csv')

# Merge the event and pick information
cat = pd.merge(events_1d, assignments_1d, left_on="idx", right_on="event_idx", suffixes=("", "_pick"))
print(cat)
print(cat.to_string())
cat.to_csv('catalog-phases-pyocto-gpd-2018.csv')

from easyQuake import pytocto_file_quakeml
from easyQuake import magnitude_quakeml
from easyQuake import simple_cat_df

# Build a QuakeML catalog from the merged CSV
file = 'catalog-phases-pyocto-gpd-2018.csv'
cat = pytocto_file_quakeml(file)

# Estimate magnitudes and write the QuakeML catalog to disk
cat = magnitude_quakeml(cat=cat, project_folder=project_folder, plot_event=False)
cat.write('catalog-2018-gpd-pyocto.xml', format='QUAKEML')

catdf = simple_cat_df(cat)
#plt.figure()
#plt.plot(catdf.index,catdf.magnitude,'.')
print(cat.__str__(print_all=True))
print(catdf)
```
### Locate with Hypoinverse
Hypoinverse is a widely used and reliable earthquake location program that applies a least-squares method with a 1D local velocity model. It refines the preliminary locations obtained from the associator, providing more accurate hypocenter estimates.
```
from easyQuake import locate_hyp2000, cut_event_waveforms, fix_picks_catalog, simple_cat_df
from obspy import read_events

project_code = '2018'
project_folder = '/data/DRproject/2018'

cat = read_events('/data/DRproject/2018/catalog-2018-gpd-pyocto.xml')

cat1 = locate_hyp2000(cat=cat, project_folder=project_folder, vel_model='VM.crh')
cat1.write('catalog-2018-gpd-pyocto-hyp.xml', format='QUAKEML')
print(cat.__str__(print_all=True))
print(cat)

#print out uncertainty parameters
catdf = simple_cat_df(cat1,True)
print(catdf.to_string())
#print(catdf)

```
### Additional Utilities
`fix_picks_catalog`: Ensures picks in a QuakeML catalog match available MiniSEED waveform files and corrects channel codes (e.g., mismatched horizontal components). This keeps the catalog consistent with stored waveforms.
```
cat2 = fix_picks_catalog(catalog=cat1, project_folder=project_folder, filename='catalog-2018-gpd-pyocto-hyp-fixed.xml'
```

`cut_event_waveforme`: Extracts waveform segments around each event in a catalog. Optionally filters and plots them, and saves the data (and figures) in an 'events/' subdirectory. Useful for catalog validation, training datasets, or manual review.

```
cut_event_waveforme(catalog=cat2, project_folder=project_folder, length=120, filteryes=True, plotevent=True)
```
### Merging ML Catalogs
We generated three separate seismic catalogs for Hispaniola using the deep-learning pickers EQTransformer, GPD, and PhaseNet within easyQuake. To build an aggregated catalog, events were matched across catalogs if their origin times differed by less than 5 seconds and their epicenters were within 200 km. When multiple matches were found, the best event was chosen based on the highest number of associated seismic phases and a realistic depth (>2 km). Unique detections from each picker were also retained, ensuring the final aggregated catalog combines common and model-specific events into a more complete ML-derived earthquake catalog.

```
from obspy import read_events, Catalog
from obspy.core.event import Event
from obspy.geodetics import gps2dist_azimuth
import numpy as np
from easyQuake import simple_cat_df

# Load catalogs
cat_pnet = read_events("catalog-pyocto-PNET-reduced.xml")
cat_gpd = read_events("catalog-pyocto-gpd-reduced.xml")
cat_eqt = read_events("catalog-pyocto-eqtransformer-reduced.xml")

time_tolerance = 5  # seconds
distance_tolerance_km = 200  # kilometers

# Get origin timestamp
def get_timestamp(event):
    return event.preferred_origin().time.timestamp

# Count number of picks (phases)
def count_phases(event):
    return len(event.preferred_origin().arrivals)

# Get distance between two events in kilometers
def get_distance_km(ev1, ev2):
    try:
        orig1 = ev1.preferred_origin()
        orig2 = ev2.preferred_origin()
        lat1, lon1 = orig1.latitude, orig1.longitude
        lat2, lon2 = orig2.latitude, orig2.longitude
        distance_m, _, _ = gps2dist_azimuth(lat1, lon1, lat2, lon2)
        return distance_m / 1000.0
    except Exception:
        return np.inf  # if location is missing, treat as infinite distance

# Choose the best event based on number of phases and depth condition
def choose_best_event(events):
    filtered = []
    for ev in events:
        try:
            depth = ev.preferred_origin().depth
            if depth is not None and 2000 <= depth <= 200000:
                filtered.append(ev)
        except Exception:
            continue
    if filtered:
        return max(filtered, key=count_phases)
    else:
        return max(events, key=count_phases)

# Create timestamp arrays
timestamps_pnet = np.array([get_timestamp(ev) for ev in cat_pnet])
timestamps_gpd = np.array([get_timestamp(ev) for ev in cat_gpd])
timestamps_eqt = np.array([get_timestamp(ev) for ev in cat_eqt])

# Output catalogs
unique_pnet = Catalog()
unique_gpd = Catalog()
unique_eqt = Catalog()
master_catalog = Catalog()

# Track matches
matched_gpd = set()
matched_eqt = set()
matched_pnet = set()

# Counters
common_all_three = 0
common_pnet_gpd = 0
common_pnet_eqt = 0
common_gpd_eqt = 0

# Match PNET events
for i, ev_pnet in enumerate(cat_pnet):
    time_pnet = timestamps_pnet[i]

    gpd_matches = [
        j for j, ev_gpd in enumerate(cat_gpd)
        if abs(timestamps_gpd[j] - time_pnet) <= time_tolerance and get_distance_km(ev_pnet, ev_gpd) <= distance_tolerance_km
    ]
    eqt_matches = [
        k for k, ev_eqt in enumerate(cat_eqt)
        if abs(timestamps_eqt[k] - time_pnet) <= time_tolerance and get_distance_km(ev_pnet, ev_eqt) <= distance_tolerance_km
    ]

    matched_gpd_flag = len(gpd_matches) > 0
    matched_eqt_flag = len(eqt_matches) > 0

    if matched_gpd_flag and matched_eqt_flag:
        common_all_three += 1
        matched_pnet.add(i)
        matched_gpd.update(gpd_matches)
        matched_eqt.update(eqt_matches)
        candidates = [ev_pnet] + [cat_gpd[j] for j in gpd_matches] + [cat_eqt[k] for k in eqt_matches]
        best_event = choose_best_event(candidates)
        master_catalog.append(best_event)
    elif matched_gpd_flag:
        common_pnet_gpd += 1
        matched_pnet.add(i)
        matched_gpd.update(gpd_matches)
        candidates = [ev_pnet] + [cat_gpd[j] for j in gpd_matches]
        best_event = choose_best_event(candidates)
        master_catalog.append(best_event)
    elif matched_eqt_flag:
        common_pnet_eqt += 1
        matched_pnet.add(i)
        matched_eqt.update(eqt_matches)
        candidates = [ev_pnet] + [cat_eqt[k] for k in eqt_matches]
        best_event = choose_best_event(candidates)
        master_catalog.append(best_event)
    else:
        unique_pnet.append(ev_pnet)
        master_catalog.append(ev_pnet)

# GPD unique or matched with EQT
for j, ev_gpd in enumerate(cat_gpd):
    if j in matched_gpd:
        continue
    time_gpd = timestamps_gpd[j]
    eqt_matches = [
        k for k, ev_eqt in enumerate(cat_eqt)
        if abs(timestamps_eqt[k] - time_gpd) <= time_tolerance and get_distance_km(ev_gpd, ev_eqt) <= distance_tolerance_km
    ]
    if len(eqt_matches) > 0:
        common_gpd_eqt += 1
        matched_eqt.update(eqt_matches)
        candidates = [ev_gpd] + [cat_eqt[k] for k in eqt_matches]
        best_event = choose_best_event(candidates)
        master_catalog.append(best_event)
    else:
        unique_gpd.append(ev_gpd)
        master_catalog.append(ev_gpd)

# EQT unique
for k, ev_eqt in enumerate(cat_eqt):
    if k not in matched_eqt:
        unique_eqt.append(ev_eqt)
        master_catalog.append(ev_eqt)

# Save catalogs
def save_catalog(catalog, base_name):
    xml_name = f"{base_name}.xml"
    csv_name = f"{base_name}.csv"
    catalog.write(xml_name, format="QUAKEML")
    df = simple_cat_df(catalog, True)
    df.to_csv(csv_name)
    print(f"Saved {xml_name} and {csv_name}")

save_catalog(unique_pnet, "unique-PNET-events-best")
save_catalog(unique_gpd, "unique-GPD-events-best")
save_catalog(unique_eqt, "unique-EQT-events-best")
save_catalog(master_catalog, "ML-master-catalog-best")

# Statistics
print("Match statistics:")
print(f"Total events in PNET: {len(cat_pnet)}")
print(f"Total events in GPD: {len(cat_gpd)}")
print(f"Total events in EQT: {len(cat_eqt)}")
print(f"Unique PNET events: {len(unique_pnet)}")
print(f"Unique GPD events: {len(unique_gpd)}")
print(f"Unique EQT events: {len(unique_eqt)}")
print(f"Total events in master catalog: {len(master_catalog)}")
print(f"Common events in all 3 catalogs: {common_all_three}")
print(f"Common events between PNET & GPD only: {common_pnet_gpd}")
print(f"Common events between PNET & EQT only: {common_pnet_eqt}")
print(f"Common events between GPD & EQT only: {common_gpd_eqt}")
```





