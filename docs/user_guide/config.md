# Config File

The following is documentation for a catchment configuration file in topoflow-glacier.

### Example Config

```yaml
site_prefix: cat-3062920
forcing_file: data/cat-3062920.csv
dt: 1
start_time: "2013032000"
end_time: "2014032000"
da: 11.418749923500716
slope: 88.582729
aspect: 242.8644693769529
lon: -121.81418
lat: 46.81953220
elev: 2446.3922737596167  # meters
h_active_layer: 0.125
h0_snow: 0.02  # related to SWE through density
h0_ice: 2.0  # related to IWE through density
h0_swe: 0.001
h0_iwe: 1.834
T_rain_snow: 0.0
```

#### Variables

###### site_prefix
The catchment from the hydrofabric

###### forcing_file
The forcing file location

###### dt
The timestep (hours)

###### start_time
The start time. This is used in determining the sun's location for snow/ice melt

###### end_time
The end time. This is used in determining the sun's location for snow/ice melt

###### da
The catchment drainage area [km2]

###### slope
Mean slope of the catchment [m/m]

###### aspect
Mean Aspect of the catchment

###### lon
Centroid Longitude Point of the Catchment

###### lat
Centroid Latitude Point of the Catchment

###### elev
Mean catchment elevation (m)

###### h_active_layer
The height of the active layer of the ice layer

###### h0_snow
The height of the snow (related to IWE through density) (m)

###### h0_ice
The height of the ice (related to IWE through density) (m)

###### h0_swe
The initial height of ice water equivalent (m)

###### h0_iwe
The initial height of ice water equivalent (m)

###### T_rain_snow
Air temperature that the precip converts to snow (degree C)
