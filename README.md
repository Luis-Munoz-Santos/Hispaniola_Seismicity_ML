# Hispaniola_seismicity_Ml
Unified machine-learning derived earthquake catalog for Hispaniola, combining regional and local datasets. Includes Python scripts, catalogs, and supplemental materials used in the study “Remarkable crustal and slab seismicity in Hispaniola revealed through a unified machine-learning derived earthquake catalog.”

## Requirements
This codes use the easyQuake python package (https://github.com/jakewalter/easyQuake) which requires

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
### Association


