import numpy as np
import pandas as pd
from pyprojroot import here

from topoflow_glacier import BmiTopoflowGlacier, __version__, configure_logging, logger


def run_topoflow_glacier():
    """The main function for running topoflow glacier"""
    configure_logging()
    logger.info(f"Running Topoflow-Glacier version: {__version__}")

    bmi_cfg_file = here() / "config/cat-3062920.yaml"

    logger.debug("Creating an instance of an BMI_LSTM model object")
    model = BmiTopoflowGlacier()

    logger.debug("Initializing the BMI")
    model.initialize(bmi_cfg_file)

    logger.debug("Gathering input data")
    df = pd.read_csv(here() / model.cfg.forcing_file)

    logger.debug("Loop through the inputs, set the forcing values, and update the model...")
    precip_data = df["RAINRATE"].values
    temp_data = df["T2D"].values
    long_wave_radiation = df["LWDOWN"].values
    short_wave_radiation = df["SWDOWN"].values
    air_pressure = df["PSFC"].values
    air_water_vapor = df["Q2D"].values  # specific humidity
    wind_speed = (
        ((df["U2D"]) ** 2 + (df["V2D"]) ** 2) ** 0.5
    ).values  # making one single wind speed based on U and V directions

    output_snow_melt = np.zeros(len(precip_data))
    output_ice_melt = np.zeros(len(precip_data))
    output_h_swe = np.zeros(len(precip_data))
    output_h_iwe = np.zeros(len(precip_data))
    output_h_snow = np.zeros(len(precip_data))
    output_h_ice = np.zeros(len(precip_data))
    output_m_total = np.zeros(len(precip_data))

    for i in range(len(precip_data)):
        model.set_value(
            "atmosphere_water__liquid_equivalent_precipitation_rate", precip_data[i] * 10 ** (-3)
        )  # converting mm/hr to m/hr
        model.set_value("land_surface_air__temperature", model.K_to_C + temp_data[i])  # converting to Celcius
        model.set_value("land_surface_radiation~incoming~longwave__energy_flux", long_wave_radiation[i])
        model.set_value("land_surface_radiation~incoming~shortwave__energy_flux", short_wave_radiation[i])
        model.set_value("land_surface_air__pressure", air_pressure[i])
        model.set_value("atmosphere_air_water~vapor__relative_saturation", air_water_vapor[i])
        model.set_value("wind_speed_UV", wind_speed[i])

        model.update()

        model.get_value("land_surface_water__runoff_depth", runoff_depth_m_hr[ts : ts + 1])

    # Finalizing the BMI
    logger.debug("Finalizing the BMI...")
    model.finalize()
    logger.debug("Finished.")


if __name__ == "__main__":
    run_topoflow_glacier()
