from pathlib import Path

import pandas as pd

from topoflow_glacier import BmiTopoflowGlacier


def run_topoflow_glacier():
    """The main function for running topoflow glacier"""
    bmi_cfg_file = Path.cwd() / "config/cat-2062920.yaml"

    print("Creating an instance of an BMI_LSTM model object")
    model = BmiTopoflowGlacier()

    print("Initializing the BMI")
    model.initialize(bmi_cfg_file)

    print("Gathering input data")
    df = pd.read_csv(model.cfg.forcing_file)

    print("Loop through the inputs, set the forcing values, and update the model...")
    precip_data = df["RAINRATE"].values
    temp_data = df["T2D"].values
    long_wave_radiation = df["LWDOWN"].values
    short_wave_radiation = df["SWDOWN"].values
    air_pressure = df["PSFC"].values
    air_water_vapor = df["Q2D"].values

    for i in range(len(precip_data)):
        model.set_value("atmosphere_water__liquid_equivalent_precipitation_rate", precip_data[i])
        model.set_value("land_surface_air__pressure", air_water_vapor[i])
        model.set_value("land_surface_radiation~incoming~longwave__energy_flux", long_wave_radiation[i])
        model.set_value("land_surface_radiation~incoming~shortwave__energy_flux", short_wave_radiation[i])
        model.set_value("land_surface_air__pressure", air_pressure[i])
        model.set_value("land_surface_air__temperature", temp_data[i])

        model.update()

    # Finalizing the BMI
    print("Finalizing the BMI...")
    model.finalize()
    print("Finished.")


if __name__ == "__main__":
    run_topoflow_glacier()
