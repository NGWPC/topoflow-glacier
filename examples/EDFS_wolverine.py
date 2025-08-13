import numpy as np

from topoflow_glacier.glacier_energy_balance import glacier_component
from topoflow_glacier.met_base_NGWPC import met_component

WV_met = met_component()
# inputs we need for meteorology
WV_met.set_constants()
WV_met.P = np.array([0.3])
WV_met.T_air = np.array([4.0])  # bottom air temperature
WV_met.T_surf = np.array([5])  # T_surf = land_surface temperature
WV_met.RH = np.array([0.7])
WV_met.p0 = np.array([0.7])  # atm pressure mbar
WV_met.z = np.array([2.0])  # the height the wind is read
WV_met.uz = np.array([10.0])  # wid speed at height z
WV_met.cloud_factor = np.array([0.10])
WV_met.canopy_factor = np.array([0.01])
WV_met.z0_air = np.array([0.0015])  # surface roughness length scale

WV_met.lat_deg = 61.5
# WV_met.lon_deg = 150.8


# albedo:
WV_met.h0_snow = np.array([2.0])
WV_met.h0_ice = np.array([2.0])
WV_met.h_snow = WV_met.h0_snow
WV_met.h_ice = WV_met.h0_ice
WV_met.albedo = np.array([0.3])  # just an initial value
WV_met.update_albedo(method="aging")  # aging or simple

WV_met.update_net_shortwave_radiation()  # gives self.Qn_SW

WV_met.net_longwave_radiation()  # gives Qn_LW

WV_glacier = glacier_component()
### inputs we need for glacier
WV_glacier.set_constants()

WV_glacier.T0 = np.array([-0.2])
WV_glacier.dt = np.array([3600])
WV_glacier.rho_snow = np.array([300])
WV_glacier.rho_ice = np.array([917])
WV_glacier.h_active_layer = np.array([0.125])
WV_glacier.h0_snow = np.array([2.0])
WV_glacier.h0_ice = np.array([2.0])
WV_glacier.h0_swe = np.array([0.6])
WV_glacier.iwe = np.array([1.834])
WV_glacier.T_surf = WV_met.T_surf  # T_surf = land_surface temperature

WV_glacier.initialize_snow_cold_content()
WV_glacier.initialize_ice_cold_content()
WV_glacier.update_snow_meltrate()
WV_glacier.update_ice_meltrate()
WV_glacier.enforce_max_snow_meltrate()
## building the glacier energy balance


print("end")
