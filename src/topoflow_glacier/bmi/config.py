from pydantic import BaseModel, ConfigDict, Field

__all__ = ["TopoflowGlacierConfig"]


class TopoflowGlacierConfig(BaseModel):
    """Validates the topoflow glacier config file"""

    # Required configuration fields
    model_config = ConfigDict(arbitrary_types_allowed=True)
    site_prefix: str = Field(description="File prefix for the study site")
    forcing_file: str = Field(description="The forcing .csv file to be used")
    n_steps: int = Field(description="Number of time steps")
    dt: float = Field(description="Timestep for snowmelt process [sec]")
    start_time: str = Field(description="The start time for the model run [YYYYMMDDHH]")
    end_time: str = Field(description="The end time for the model run [YYYYMMDDHH]")
    da: float = Field(description="The drainage area of the modeled reach [m2]")
    slope: float = Field(description="The slope of the catchment [-]")
    lat: float = Field(description="Latitude of the centroid of the catchment")
    lon: float = Field(description="Longitude of the centroid of the catchment")
    h0_snow: float = Field(description="Initial depth of snow [m]")
    h0_ice: float = Field(description="Initial depth of ice [m]")
    h0_swe: float = Field(description="Initial depth of snow water equivalent (SWE) [m]")
    h0_iwe: float = Field(description="Initial depth of ice water equivalent (IWE) [m]")

    rho_snow: float = Field(50.0, description="Density of snow [kg/m^3]")
    rho_ice: float = Field(917.0, description="Density of ice [kg/m^3]")
    rho_H2O: float = Field(1000.0, description="Density of water [kg/m^3]")
    h_active_layer: float = Field(0.125, description="Thickness of active ice layer [m]")
    T0: float = Field(-0.2, description="Reference temperature [deg C]")
    Cp_ice: float = Field(2060.0, description="Specific heat capacity of ice [J/(kg * K)]]")
    Cp_snow: float = Field(2090.0, description="Specific heat capacity of snow [J/(kg * K)]]")
    g: float = Field(9.81, description="Gravity  [m/s2]")
    Lf: float = Field(334000.0, description="Latent heat of fusion [J kg-1]")

    min_glacier_thick: float = Field(default=1.0, description="Minimum glacier thickness [m]")
    glens_A: float = Field(default=2.142e-16, description="Glen's Law exponent [Pa^-3 s^-1]")
    B: float = Field(default=0.0012, description="Flow law parameter [m / (Pa * yr)], see MacGregor (2000)")
    char_sliding_vel: float = Field(default=10.0, description="Characteristic sliding velocity [m/yr]")
    char_tau_bed: float = Field(default=100000.0, description="Characteristic shear stress at the bed [Pa]")
    depth_to_water_table: float = Field(
        default=20.0, description="Distance from ice surface to water table [m]"
    )
    max_float_fraction: float = Field(default=80.0, description="Limits the water level in ice [percent]")
    Hp_eff: float = Field(default=20.0, description="Effective pressure [m] of water")
    init_ELA: float = Field(default=3350.0, description="Initial Equilibrium Line Altitude [m]")
    ELA_step_size: float = Field(default=-10.0, description="ELA step size [m]")
    ELA_step_interval: float = Field(default=500.0, description="ELA step interval [m]")
    grad_Bz: float = Field(default=0.01, description="Mass balance gradient in z [m/yr/m]")
    max_Bz: float = Field(default=2.0, description="Maximum allowed mass balance [m/yr]")
    spinup_time: float = Field(default=200.0, description="Spinup time [years]")
    sea_level: float = Field(default=-100.0, description="Sea level [m]")

    geothermal_heat_flux: float = Field(default=1575000.0, description="Geothermal heat flux [(J/year)/m^2]")
    geothermal_gradient: float = Field(default=-0.0255, description="Geothermal gradient [deg_C/m]")
