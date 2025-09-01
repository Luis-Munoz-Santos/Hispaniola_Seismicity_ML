# Hispaniola_seismicity_Ml
Unified machine-learning derived earthquake catalog for Hispaniola, combining regional and local datasets. Includes Python scripts, catalogs, and supplemental materials used in the study “Remarkable crustal and slab seismicity in Hispaniola revealed through a unified machine-learning derived earthquake catalog.”

## Requirements
These codes use the easyQuake python package (https://github.com/jakewalter/easyQuake) which requires

- Python 3.7
- Tensorflow-gpu 2.2
- Obspy
- Keras

### Download data
In order to build the catalog we first need to download the raw data from repositories of the International Federation of Digital Seismograph Networks (FDSN) that are available to the public.
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

project_code = 'DR-2018'
project_folder = '/data/grasp/Easyquake/DRproject/2018'
for single_date in daterange(start_date, end_date):
    print(single_date.strftime("%Y-%m-%d"))
    dirname = single_date.strftime("%Y%m%d")
    download_mseed(dirname=dirname, project_folder=project_folder, single_date=single_date, minlat=lat_a, maxlat=lat_b, minlon=lon_a, maxlon=lon_b, raspberry_shake=True)
```
### Phase Detection
easyQuake integrate three deep-learning pickers (EQTransformer, PhaseNet, GPD) which can be selected in the detection_continuous function to perform event detection and seismic phase picking.

```
from easyQuake import detection_continuous
from easyQuake import daterange
from datetime import date
start_date = date(2018, 1, 1)
end_date = date(2019, 1, 1)

project_code = '2018'
project_folder = '/data/grasp/Easyquake/DRproject/2018'
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
This script use the default associatior withing easyQuake and the Python multiprocessing package to parallelize the job.

```
from easyQuake import association_continuous
from easyQuake import daterange
from easyQuake import combine_associated
from datetime import date
from easyQuake import magnitude_quakeml
from easyQuake import simple_cat_df
import matplotlib.pyplot as plt

start_date = date(2018, 1, 1)
end_date = date(2019, 1, 1)
maxdist = 300
maxkm = 300

from multiprocessing import Pool
pool = Pool(40)

project_code = '2018'
project_folder = '/data/grasp/Easyquake/DRproject/2018'

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

cat = magnitude_quakeml(cat=cat, project_folder=project_folder, plot_event=False)
cat.write('catalog-2018-GPD.xml',format='QUAKEML')

catdf = simple_cat_df(cat)
#plt.figure()
#plt.plot(catdf.index,catdf.magnitude,'.')

print(cat.__str__(print_all=True))
#print(catdf)
```
For using PyOcto associator (https://github.com/yetinam/pyocto) instead, you can use the following script:

```
import pyocto
import pandas as pd
import datetime
import matplotlib.pyplot as plt

#read picks from deep learning picker
picks = pd.read_csv("gpd-picks-2018-pyocto.csv")
print(picks)

#convert 'time' column to datetime format
picks["time"]=pd.to_datetime(picks["time"])
print(picks)

#convert picks from datetime to simple floats
picks["time"] = picks["time"].apply(lambda x: x.timestamp())
print(picks)

#read station information
stations = pd.read_csv("stations-list-2018-pyocto.csv")
print(stations)

#Create 1D velocity model
layers = pd.read_csv("DRMODEL.csv")
layers
model_path = "DR_velocity_model"
pyocto.VelocityModel1D.create_model(layers, 1., 400, 250, model_path)

#load the velocity model
velocity_model = pyocto.VelocityModel1D(model_path, tolerance=2.0)

#create an associator instance and run the association
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

#Convert event times and coordinate informations back to datetime format, latitude and longitude
associator.transform_events(events_1d)
events_1d["time"] = events_1d["time"].apply(datetime.datetime.fromtimestamp, tz=datetime.timezone.utc)
print(events_1d)
print(events_1d.to_string())
#pd.set_option('display.max_columns', None)
#print(events_1d)
events_1d.to_csv('catalog-pyocto-gpd-2018.csv')

#Merge the picks and event information
cat = pd.merge(events_1d, assignments_1d, left_on="idx", right_on="event_idx", suffixes=("", "_pick"))
print(cat)
print(cat.to_string())

cat.to_csv('catalog-phases-pyocto-gpd-2018.csv')





