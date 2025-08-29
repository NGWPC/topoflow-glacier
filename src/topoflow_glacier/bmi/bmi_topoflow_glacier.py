from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from topoflow_glacier.bmi.bmi_base import BmiBase
from topoflow_glacier.bmi.config import TopoflowGlacierConfig
from topoflow_glacier.physics import solar_funcs as solar
from topoflow_glacier.physics.context import build_context

__all__ = ["BmiTopoflowGlacier"]

_dynamic_input_vars = [
    ("land_surface_radiation~incoming~longwave__energy_flux", "W m-2"),
    ("land_surface_air__pressure", "Pa"),
    ("atmosphere_air_water~vapor__relative_saturation", "kg kg-1"),
    ("atmosphere_water__liquid_equivalent_precipitation_rate", "mm h-1"),
    ("land_surface_radiation~incoming~shortwave__energy_flux", "W m-2"),
    ("land_surface_air__temperature", "degK"),
]

_output_vars = [
    ("land_surface_water__runoff_volume_flux", "m3 s-1"),
]

# @FRahmani368 Please change this dictionary to map the BMI values to the var names
_var_name_map = {
    # ---------------------------------------------------------------
    "glacier_ice__domain_time_integral_of_melt_volume_flux": "vol_MR",
    "glacier_ice__melt_volume_flux": "mr_ice",
    "glacier_top_surface__elevation": "z_ice",
    "glacier_ice__thickness": "h_ice",
}
_input_alias_map = {
    "atmosphere_water__liquid_equivalent_precipitation_rate": "P",         # precip
    "land_surface_air__temperature": "T_air",                               # air temp
    # "land_surface_radiation~incoming~shortwave__energy_flux": "SW_in",    # SW down
    "land_surface_radiation~incoming~longwave__energy_flux": "LW_in",     # LW down
    "land_surface_air__pressure": "P_air",                                  # total air pressure
    "atmosphere_air_water~vapor__relative_saturation": "Hum_sp",             # specific humidity
}



class BmiTopoflowGlacier(BmiBase):
    """BMI composition wrapper for TopoflowGlacier"""

    def __init__(self) -> None:
        self._dynamic_inputs = build_context(_dynamic_input_vars)
        self._outputs = build_context(_output_vars)
        self._timestep: int = 0

    def initialize(self, config_file: str | Path) -> None:
        """Intialize the BMI model with config, datum transformer, and datum sync class.

        Args:
            config (str): _description_
        """
        # read yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)

        # load into pydantic model and save in class for querying
        self.cfg = TopoflowGlacierConfig.model_validate(config)
        self.hours_per_day = np.float64(24)
        self.seconds_per_Day = np.float64(24) * 3600
        self.sec_per_year = np.float64(3600) * 24 * 365  # [secs]
        self.mps_to_mmph = np.float64(3600000)
        self.mmph_to_mps = np.float64(1) / np.float64(3600000)
        self.dt = self.cfg.dt
        self.days_per_dt = self.dt / 86400
        self.n = 0.0   # For albedo calculations, Start 'number of days since major snowfall' at 0
        self.C_to_K = 273.15
        self.K_to_C = - 273.15
        self.twopi = np.float64(2) * np.pi
        self.one_seventh = np.float64(1) / 7

        # TODO create state variables for these
        self.P = np.array(0, dtype="float64")
        self.T_air = np.array(0, dtype="float64")  # bottom air temperature
        self.T_surf = np.array(0, dtype="float64")  # T_surf = land_surface temperature
        self.RH = np.array(0, dtype="float64")
        self.p0 = np.array(0, dtype="float64")  # atm pressure mbar
        self.z = np.array(0, dtype="float64")  # the height the wind is read
        self.uz = np.array(0, dtype="float64")  # wind speed at height z.
        self.cloud_factor = np.array(0, dtype="float64")
        self.canopy_factor = np.array(0, dtype="float64")
        self.P_rain = np.array(0, dtype="float64")
        self.P_snow = np.array(0, dtype="float64")
        self.e_air = np.array(0, dtype="float64")
        self.e_surf = np.array(0, dtype="float64")
        self.em_air = np.array(0, dtype="float64")
        # self.SW_in = np.array(0, dtype="float64")
        self.LW_in = np.array(0, dtype="float64")
        self.Qn_SW = np.array(0, dtype="float64")
        self.Qn_LW = np.array(0, dtype="float64")
        self.Q_sum = np.array(0, dtype="float64")
        self.Qc = np.array(0, dtype="float64")
        self.Qa = np.array(0, dtype="float64")
        self.P_max = np.array(0, dtype="float64")
        self.vol_P = np.array(0, dtype="float64")
        self.vol_PR = np.array(0, dtype="float64")
        self.vol_PS = np.array(0, dtype="float64")
        self.Qn_SW = np.array(0, dtype="float64")
        self.Qn_LW = np.array(0, dtype="float64")
        self.Qn_tot = np.array(0, dtype="float64")
        self.Q_sum = np.array(0, dtype="float64")
        self.Qe = np.array(0, dtype="float64")
        self.e_air = np.array(0, dtype="float64")
        self.e_surf = np.array(0, dtype="float64")
        self.em_air = np.array(0, dtype="float64")
        self.Qc = np.array(0, dtype="float64")
        self.Qa = np.array(0, dtype="float64")
        self.P_air = np.array(0, dtype="float64")
        self.Hum_sp = np.array(0, dtype="float64")
        self.P_snow_3day_watershed = np.zeros(int(3 * self.seconds_per_Day / self.cfg.dt), dtype="float64")

        # Ice component
        self.rho_H2O = np.float64(self.cfg.rho_H2O)  # [kg/m**3]
        self.rho_ice = np.float64(self.cfg.rho_ice)  # [kg/m**3]
        self.Cp_ice = np.float64(self.cfg.Cp_ice)  # [J/(kg * K)]
        self.Qg = np.float64(self.cfg.geothermal_heat_flux)  # [(J/yr)/m**2]
        self.grad_Tz = np.float64(self.cfg.geothermal_gradient)  # [deg_C/m]
        self.g = np.float64(self.cfg.g)  # [m/s**2]

        # Glacier Component
        self.rho_snow = np.float64(self.cfg.rho_snow)  # [kg/m**3]
        self.Cp_snow = np.float64(self.cfg.Cp_snow)  # [J kg-1 K-1]
        self.Lf = np.float64(self.cfg.Lf)  # [J kg-1]
        self.T0 = np.array(self.cfg.T0, dtype="float64")  # [deg C]
        self.h_active_layer = np.array(self.cfg.h_active_layer, dtype="float64")  # [m]
        self.T_rain_snow = np.float64(self.cfg.T_rain_snow)

        self.h_snow = np.array(self.cfg.h0_snow, dtype="float64")  # [m]
        self.h_ice = np.array(self.cfg.h0_ice, dtype="float64")  # [m]
        self.h_swe = np.array(self.cfg.h0_swe, dtype="float64")  # [m]
        self.h_iwe = np.array(self.cfg.h0_iwe, dtype="float64")  # [m]
        self.mr_ice = np.array(0, dtype="float64")
        self.vol_MR = np.array(0, dtype="float64")
        self.meltrate = np.array(0, dtype="float64")

        # Glacier Component
        self.SM = np.array(0, dtype="float64")
        self.IM = np.array(0, dtype="float64")
        self.M_total = np.array(0, dtype="float64")
        self.vol_SM = np.array(0, dtype="float64")  # [m3]
        self.vol_IM = np.array(0, dtype="float64")
        self.vol_M_total = np.array(0, dtype="float64")
        self.vol_swe = np.array(0, dtype="float64")  # [m3]
        self.vol_swe_start = np.array(0, dtype="float64")
        self.vol_iwe = np.array(0, dtype="float64")
        self.vol_iwe_start = np.array(0, dtype="float64")
        self.albedo = np.array([0.3])

        # Update Snowpack Water Volume
        volume = np.float64(self.h_swe * self.cfg.da)  # [m^3]
        vol_swe = np.sum(volume)
        self.vol_swe.fill(vol_swe)
        self.vol_IM = np.array(0, dtype="float64")  # (m3)

        # Update Ice Water Equivalent
        volume = np.float64(self.h_iwe * self.cfg.da)  # [m^3]
        vol_iwe = np.sum(volume)
        self.vol_iwe.fill(vol_iwe)

        self.vol_M_total = np.array(0, dtype="float64")
        self.ws_density_ratio = self.rho_H2O / self.rho_snow
        self.wi_density_ratio = self.rho_H2O / self.rho_ice

        self.T0_cc = self.T0  # synonyms
        T_snow = self.T_surf
        del_T = self.T0_cc - T_snow
        self.Eccs = (self.rho_snow * self.Cp_snow) * self.h_snow * del_T
        self.Eccs = np.maximum(self.Eccs, 0.0)
        self.Ecci = (self.rho_ice * self.Cp_ice) * self.h_active_layer * del_T
        self.Ecci = np.maximum(self.Ecci, 0.0)

        self.start_year, self.start_month, self.start_day, self.start_hour = self._parse_yyyymmddhh(
            self.cfg.start_time
        )
        self.end_year, self.end_month, self.end_day, self.end_hour = self._parse_yyyymmddhh(self.cfg.end_time)
        self.year = self.start_year
        self.julian_day = solar.Julian_Day(
            self.start_month, self.start_day, self.start_hour, year=self.start_year
        )
        self.start_datetime = pd.to_datetime(
                solar.get_datetime_str(
                    self.start_year,
                    self.start_month,
                    self.start_day,
                    self.start_hour, 0, 0
            )
        )

    def update(self) -> None:
        """Update the model based on inputs (only meterological and glacier currently)"""
        self.update_atm_pressure_from_elevation(T_C = True, MBAR=True)
        # Update Meteorological Component
        self.update_P_integral()  # update vol_P (leq)
        self.update_P_max()
        self.update_P_rain()
        self.update_P_snow()
        self.update_P_rain_integral()  # update vol_PR
        self.update_P_snow_integral()  # update vol_PS (leq)
        self.update_saturation_vapor_pressure(MBAR=True)
        self.update_vapor_pressure_from_spHum_AirPre(MBAR=True)
        self.update_RH()
        # self.update_vapor_pressure()
        self.update_dew_point()  ###
        self.update_T_surf()
        self.update_saturation_vapor_pressure(MBAR=True, SURFACE=True)  ########
        self.update_bulk_richardson_number()
        self.update_bulk_aero_conductance()
        self.update_sensible_heat_flux()
        self.update_precipitable_water_content()  ###
        self.update_vapor_pressure(SURFACE=True)  ########
        self.update_latent_heat_flux()  # (uses e_air and e_surf)
        self.update_conduction_heat_flux()    # currently assumed zero
        self.update_advection_heat_flux()     # currently assumed zero
        self.update_julian_day()
        self.update_albedo(method="aging")
        self.set_aspect_angle()
        self.set_slope_angle()
        self.update_net_shortwave_radiation()
        self.update_em_air()
        self.update_net_longwave_radiation()
        self.update_net_energy_flux()  # (at the end)

        # Update Glacier component
        self.extract_previous_swe()
        self.extract_previous_snow_depth()
        self.update_snow_meltrate()  # (meltrate = SM)
        self.enforce_max_snow_meltrate()  # (before SM integral!)
        self.update_SM_integral()
        self.update_swe()
        self.update_snowfall_cold_content()
        self.update_ice_meltrate()
        self.enforce_max_ice_meltrate()
        self.update_IM_integral()
        self.update_combined_meltrate()

        self.update_iwe()  # relies on previous timestep's swe value
        self.update_density_ratio()
        self.update_snow_depth()
        self.update_ice_depth()
        self.update_snowpack_cold_content()

    def finalize(self) -> None:
        """Clean up any internal resources of the model"""
        pass

    def _parse_yyyymmddhh(self, s: str) -> tuple[int, int, int, int]:
        """Accepts 'YYYYMMDD-HH' (e.g., '20231001-01') or 'YYYYMMDDHH'. Returns (year, month, day, hour, dt)."""
        s = str(s).strip()
        fmt = "%Y%m%d-%H" if "-" in s else "%Y%m%d%H"
        dt = datetime.strptime(s, fmt)  # raises ValueError if malformed
        return dt.year, dt.month, dt.day, dt.hour

    def update_atm_pressure_from_elevation(self, T_C=True, MBAR=False):
        """
        Estimate atmospheric pressure at elevation z_m (meters).

        Parameters
        ----------
        z_m : float
            Elevation above sea level [m].
        T_C : float, optional
            Air temperature [°C] for the isothermal exponential model.
            If None, uses the standard atmosphere formula with lapse rate.
        MBAR : bool, optional
            If True, return pressure in hPa (mbar). Default False = Pa.

        Returns
        -------
        p0 : float
            Atmospheric pressure [Pa]
        """
        # constants
        sea_level_p0 = self.cfg.sea_level_p0  # sea-level standard pressure [Pa]
        T0 = self.cfg.sea_level_T0  # sea-level standard temperature [K]
        g = self.cfg.g  # gravity [m/s2]
        L = self.cfg.T_lapse_rate  # temperature lapse rate [K/m]
        R_star = self.cfg.uni_gas_const  # universal gas constant [J/mol/K]
        M = self.cfg.M_mass_air  # molar mass of dry air [kg/mol]

        if T_C == False:
            # Standard atmosphere with lapse rate
            self.p0 = sea_level_p0 * (1 - (L * self.cfg.elev) / T0) ** (g * M / (R_star * L))  # Pa
        else:
            # Isothermal assumption with given T in Celsius
            T_K = self.T_air + 273.15
            self.p0 = sea_level_p0 * np.exp(-M * g * self.cfg.elev / (R_star * T_K))  # Pa
        self.p0 = self.p0 / np.float64(1000)  # [kPa]

        if MBAR:  # KPa to kpa
            self.p0 = self.p0 * np.float64(10.0)

    def update_P_integral(self):
        """Update mass total for P, sum over all pixels
        -------------------------------------------------
        We need to include total precip here, that is,
        P = P_rain + P_snow (liquid equivalent), not
        just P_rain, for use in a mass balance check.
        P_rain and da are both either scalar or grid.
        -------------------------------------------------
        """  # noqa: D205
        volume = np.double(self.P * self.cfg.da * self.dt)  # [m^3]
        self.vol_P += np.sum(volume)

    def update_P_max(self):
        """Save the maximum precip. rate in [m/s]
        -------------------------------------------
        Must use "fill()" to preserve reference.
        -------------------------------------------
        """  # noqa: D205
        self.P_max.fill(np.maximum(self.P_max, self.P.max()))

    def update_P_rain(self):
        """P_rain is the precip that falls as liquid that
        can contribute to runoff production.
        -------------------------------------------------
        P_rain is used by channel_base.update_R.
        -------------------------------------------------
        """  # noqa: D205
        P_rain = self.P * (self.T_air > self.T_rain_snow)

        if np.ndim(self.P_rain) == 0:
            self.P_rain.fill(P_rain)  #### (mutable scalar)
        else:
            self.P_rain = P_rain

    def update_P_snow(self):
        """P_snow is the precip that falls as snow or ice
        that contributes to the snow depth.  This snow
        may melt to contribute to runoff later on.
        -------------------------------------------------
        P_snow is a "water equivalent" volume flux
        that was determined from a total volume flux
        and a rain-snow temperature threshold.
        -------------------------------------------------
        P_snow is used by snow_base.update_depth.
        -------------------------------------------------
        """  # noqa: D205
        self.P_snow = self.P * (self.T_air <= self.T_rain_snow)

    def update_P_rain_integral(self):
        """Update mass total for P, sum over all pixels
        ------------------------------------------------
        2023-08-31. This one only uses P_rain.
        P_rain and da are both either scalar or grid.
        ------------------------------------------------
        """  # noqa: D205
        volume = np.double(self.P_rain * self.cfg.da * self.dt)  # [m^3]
        self.vol_PR += np.sum(volume)

    def update_P_snow_integral(self):
        """Update mass total for P_snow, sum over all pixels
        # ----------------------------------------------------
        # 2023-09-11. This one only uses P_snow.
        # P_snow and da are both either scalar or grid.
        # ------------------------------------------------
        """  # noqa: D205
        volume = np.double(self.P_snow * self.cfg.da * self.dt)  # [m^3]
        self.vol_PS += np.sum(volume)

    def update_bulk_richardson_number(self):
        """
        (9/6/14)  Found a typo in the Zhang et al. (2000) paper,
        in the definition of Ri.  Also see Price and Dunne (1976).
        We should have (Ri > 0) and (T_surf > T_air) when STABLE.
        This also removes problems/singularities in the corrections
        for the stable and unstable cases in the next function.
        ---------------------------------------------------------------
        Notes: Other definitions are possible, such as the one given
               by Dingman (2002, p. 599).  However, this one is the
               one given by Zhang et al. (2000) and is meant for use
               with the stability criterion also given there.
        ---------------------------------------------------------------
        """  # noqa: D205
        top = self.g * self.z * (self.T_air - self.T_surf)
        bot = (self.uz) ** 2.0 * (self.T_air + np.float64(273.15))
        if bot == 0.0:
            bot = 0.01   # to prevent denominator becomes zero
        self.Ri = top / bot

    def update_bulk_aero_conductance(self):
        """Notes: Dn       = bulk exchange coeff for the conditions of
                          neutral atmospheric stability [m/s]
               Dh       = bulk exchange coeff for heat  [m/s]
               De       = bulk exchange coeff for vapor [m/s]
               h_snow   = snow depth [m]
               z0_air   = surface roughness length scale [m]
                          (includes vegetation not covered by snow)
               z        = height that has wind speed uz [m]
               uz       = wind speed at height z [m/s]
               kappa    = 0.408 = von Karman's constant [unitless]
               RI       = Richardson's number (see function)
        ----------------------------------------------------------------
        Compute bulk exchange coeffs (neutral stability)
        using the logarithm "law of the wall".
        -----------------------------------------------------
        Note that "arg" = the drag coefficient (unitless).
        -----------------------------------------------------
        Dn will be a grid if any of the variables:
          z, h_snow, z0_air, or uz is a grid.
        -----------------------------------------------------
        """  # noqa: D205
        h_snow = self.h_snow  # (ref from new framework)

        arg = self.cfg.kappa / np.log(np.maximum((self.z - h_snow) / self.cfg.z0_air, 0.01))
        Dn = self.uz * (arg) ** 2.0
        if self.T_air == self.T_surf:
            nw = 0
        else:
            nw = 1

        if nw == 0:
            # --------------------------------------------
            # All pixels are neutral. Set Dh = De = Dn.
            # --------------------------------------------
            self.Dn = Dn
            self.Dh = Dn
            self.De = Dn
            return

        Dh = Dn.copy()  ### (9/7/14.  Save Dn also.)
        nD = Dh.size
        nR = self.Ri.size
        if nR > 1:
            # --------------------------
            # Case where RI is a grid
            # --------------------------
            ws = self.Ri > 0  # where stable
            ns = ws.sum()
            wu = np.invert(ws)  # where unstable
            nu = wu.sum()

            if nD == 1:
                # -----------------------------------
                # Convert Dh to a grid, same as Ri
                # -----------------------------------
                Dh = Dh + np.zeros(self.Ri.shape, dtype="float64")

            # ----------------------------------------------------------
            # If (Ri > 0), or (T_surf > T_air), then STABLE. (9/6/14)
            # ----------------------------------------------------------
            # When ws and wu are boolean arrays, don't
            # need to check whether any are True.
            # -------------------------------------------
            # Dh[ws] = Dh[ws] / (np.float64(1) + (np.float64(10) * self.Ri[ws]))
            # Dh[wu] = Dh[wu] * (np.float64(1) - (np.float64(10) * self.Ri[wu]))
            # -----------------------------------------------------------------------
            if ns != 0:
                Dh[ws] = Dh[ws] / (np.float64(1) + (np.float64(10) * self.Ri[ws]))
            if nu != 0:
                Dh[wu] = Dh[wu] * (np.float64(1) - (np.float64(10) * self.Ri[wu]))
        else:
            # ----------------------------
            # Case where Ri is a scalar
            # --------------------------------
            # Works if Dh is grid or scalar
            # --------------------------------
            if self.Ri > 0:
                Dh = Dh / (np.float64(1) + (np.float64(10) * self.Ri))
            else:
                Dh = Dh * (np.float64(1) - (np.float64(10) * self.Ri))

        # ----------------------------------------------------
        # NB! We currently assume that these are all equal.
        # ----------------------------------------------------
        self.Dn = Dn
        self.Dh = Dh
        self.De = Dh  ## (assumed equal)

    def update_sensible_heat_flux(self):
        """Physical constants
        ---------------------
        rho_air = 1.225d   ;[kg m-3, at sea-level]
        Cp_air  = 1005.7   ;[J kg-1 K-1]
        -----------------------------
        Compute sensible heat flux
        -----------------------------
        """  # noqa: D205
        delta_T = self.T_air - self.T_surf
        self.Qh = (self.cfg.rho_air * self.cfg.Cp_air) * self.Dh * delta_T

    def update_saturation_vapor_pressure(self, MBAR=False, SURFACE=False):
        """Notes: Saturation vapor pressure is a function of temperature.
        #        T is temperature in Celsius.  By default, the method
        #        of Brutsaert (1975) is used, but if the SATTERLUND
        #        keyword is set then the method of Satterlund (1979) is
        #        used.  When plotted, they look almost identical.  See
        #        the compare_em_air_method routine in this file.
        #        Dingman (2002) uses the Brutsaert method.
        #        Liston (1995, EnBal) uses the Satterlund method.

        #        By default, the result is returned with units of kPa.
        #        Set the MBAR keyword for units of millibars.
        #        100 kPa = 1 bar = 1000 mbars
        #                => 1 kPa = 10 mbars
        # ----------------------------------------------------------------
        # NB!    Here, 237.3 is correct, and not a misprint of 273.2.
        #        See footnote on p. 586 in Dingman (Appendix D).
        # ----------------------------------------------------------------
        # Also see: topoflow.utils.met_utils.py   #################
        # ----------------------------------------------------------------
        # NOTE: If the temperature, T_air or T_surf, is constant in
        #       time, so that T_air_type or T_surf_type is in
        #       ['Scalar', 'Grid'], and if it has been initialized
        #       correctly, then there is no need to recompute e_sat.
        # ----------------------------------------------------------------
        """  # noqa: D205
        if SURFACE:
            #             HAVE_VAR   = hasattr(self, 'e_sat_surf'))
            #             T_CONSTANT = (self.T_surf_type in ['Scalar', 'Grid'])
            #             if (HAVE_VAR and T_CONSTANT): return
            T = self.T_surf
        else:
            #             HAVE_VAR   = hasattr(self, 'e_sat_air')
            #             T_CONSTANT = (self.T_air_type in ['Scalar', 'Grid'])
            #             if (HAVE_VAR and T_CONSTANT): return
            T = self.T_air

        if not (self.cfg.SATTERLUND):
            # ------------------------------
            # Use Brutsaert (1975) method
            # ------------------------------
            term1 = (np.float64(17.3) * T) / (T + np.float64(237.3))
            e_sat = np.float64(0.611) * np.exp(term1)  # [kPa]
        else:
            # -------------------------------
            # Use Satterlund (1979) method     #### DOUBLE CHECK THIS (7/26/13)
            # -------------------------------
            term1 = np.float64(2353) / (T + np.float64(273.15))
            e_sat = np.float64(10) ** (np.float64(11.4) - term1)  # [Pa]
            e_sat = e_sat / np.float64(1000)  # [kPa]

        # -----------------------------------
        # Convert units from kPa to mbars?
        # -----------------------------------
        if MBAR:
            e_sat = e_sat * np.float64(10)  # [mbar]

        if SURFACE:
            self.e_sat_surf = e_sat
        else:
            self.e_sat_air = e_sat

    def update_vapor_pressure_from_spHum_AirPre(self, SURFACE=False, MBAR=False):
        """
        computes vapor pressure using specific humidity and total air pressure

        :param SURFACE: Flase or True
        :param MBAR: converts to mbar
        :return: None
        """

        e = self.Hum_sp * self.P_air / (self.cfg.eps + ((1 - self.cfg.eps) * self.Hum_sp))
        e = e / np.float64(1000)  # [kPa]

        if MBAR:
            e = e * np.float64(10)  # [mbar]


        if SURFACE:
            self.e_surf = e
        else:
            self.e_air = e

    def update_RH(self, SURFACE=False):
        """
        Updates relative humidity. Between [0, 1]

        :param SURFACE: False or True
        :return: None
        """

        if SURFACE:
            self.RH = self.e_surf / self.e_sat_surf
        else:
            self.RH = self.e_air / self.e_sat_air


    def update_vapor_pressure(self, SURFACE=False):
        """Notes: T is temperature in Celsius
        #        RH = relative humidity, in [0,1]
        #             by definition, it equals (e / e_sat)
        #        e has units of kPa.
        # ---------------------------------------------------
        """  # noqa: D205
        if SURFACE:
            e_sat = self.e_sat_surf
        else:
            e_sat = self.e_sat_air

        # -------------------------------------------------
        e = self.RH * e_sat

        if SURFACE:
            self.e_surf = e
        else:
            self.e_air = e

    def update_dew_point(self):
        """Notes:  The dew point is a temperature in degrees C and
        #         is a function of the vapor pressure, e_air.
        #         Vapor pressure is a function of air temperature,
        #         T_air, and relative humidity, RH.
        # -----------------------------------------------------------

        # -------------------------------------------
        # This formula needs e_air in kPa units.
        # See: Dingman (2002, Appendix D, p. 587).
        # 2023-09-01.  But it may contain a bug.
        # -------------------------------------------
        #         e_air_kPa = self.e_air / np.float64(10) # [mbar -> kPa]
        #         log_vp    = np.log( e_air_kPa )
        #         top = log_vp + np.float64(0.4926)
        #         bot = np.float64(0.0708) - (np.float64(0.00421) * log_vp)
        #         self.T_dew = (top / bot)    # [degrees C]
        # -------------------------------------------
        # This formula needs e_air in Pa units.
        # See: Dingman (2015, 3.2.5, p. 114).
        # -------------------------------------------
        #         e_air_Pa = self.e_air * 100 # [mbar -> Pa]
        #         self.T_dew = (top / bot)    # [degrees C]
        # -----------------------------------------------
        # This formula needs e_air in mbar units.
        # See: https://en.wikipedia.org/wiki/Dew_point
        # -----------------------------------------------
        """  # noqa: D205
        a = 6.1121  # [mbar]
        b = 18.678
        c = 257.14  # [deg C]
        # d = 234.5    # [deg C]
        log_term = np.log(self.e_air / a)
        self.T_dew = c * log_term / (b - log_term)  # [deg C]

    def update_T_surf(self):
        """
        # Estimate T_surf using T_dew (Raleigh et al. 2013).
        # Only run this function if T_surf is provided
        # as a scalar or grid so that it still varies in time
        # -------------------------------------------------
        """  # noqa: D205
        # -------------------------------------------------
        # If snow and/or ice are present,  T_surf cannot
        # exceed 0 deg C
        # -------------------------------------------------
        T_surf = np.where(
            ((self.h_snow > 0) | (self.h_ice > 0)),  # where snow or ice exists
            np.minimum(self.T_dew, np.float64(0)),  # T_surf is either T_dew or 0, whichever is lower
            self.T_dew,
        )  # everywhere else, T_surf = T_dew
        self.T_surf = T_surf

    def update_precipitable_water_content(self):
        """
        # Notes:  W_p is precipitable water content in centimeters,
        #         which depends on air temp and relative humidity.
        # ------------------------------------------------------------
        """  # noqa: D205
        arg = np.float64(0.0614 * self.T_dew)
        self.W_p = np.float64(1.12) * np.exp(arg)  # [cm]

    def update_latent_heat_flux(self):
        """Notes:  Pressure units cancel out because e_air and
        #         e_surf (in numer) have same units (mbar) as
        #         p0 (in denom).
        # --------------------------------------------------------
        # According to Dingman (2002, p. 273), constant should
        # be 0.622 instead of 0.662 (Zhang et al., 2000).
        # --------------------------------------------------------
        """  # noqa: D205
        const = self.cfg.latent_heat_constant
        factor = self.cfg.rho_air * self.cfg.Lv * self.De
        delta_e = self.e_air - self.e_surf
        self.Qe = factor * delta_e * (const / self.p0)

    def update_conduction_heat_flux(self):
        """Notes: The conduction heat flux from snow to soil for
        #        computing snowmelt energy, Qm, is close to zero.
        #        Currently, self.Qc = 0 in initialize().

        #        However, the conduction heat flux from surface and sub-
        #        surface for computing Qet is given by Fourier's Law,
        #        namely Qc = Ks(Tx - Ts)/x.

        #        All the Q's have units of W/m^2 = J/(m^2 s).
        # -----------------------------------------------------------------
        """  # noqa: D205
        pass  # Method not implemented in Topoflow: https://github.com/NOAA-OWP/topoflow/blob/db4d5877a32455beebe78edf5abe8d91df128665/topoflow/components/met_base.py#L1905

    def update_advection_heat_flux(self):
        """Notes: Currently, self.Qa = 0 in initialize().
        #        All the Q's have units of W/m^2 = J/(m^2 s).
        # ------------------------------------------------------
        """  # noqa: D205
        pass  # Method not implemented in Topoflow: https://github.com/NOAA-OWP/topoflow/blob/db4d5877a32455beebe78edf5abe8d91df128665/topoflow/components/met_base.py#L1925

    def update_julian_day(self, time_units="seconds"):
        """Update the julian_day and year using pandas datetime."""
    # -------------------------------------------------------
    # Compute the current datetime from start + offset
    # -------------------------------------------------------
        self.get_current_datetime(time_units=time_units)

        self.year = self.start_datetime.year


        # ----------------------------------
        # Update the *decimal* Julian day
        # ----------------------------------
        self.julian_day = (
                        self.start_datetime.day_of_year - 1
                        + self.start_datetime.hour/24
                        + self.start_datetime.minute/1440
                        + self.start_datetime.second/86400
                    )
        # print('Julian Day =', self.julian_day)

        # ----------------------------------
        # Update the *decimal* Julian day
        # --------------------------------------------------
        # Before 2021-07-29, but doesn't stay in [1,365].
        # --------------------------------------------------
        ## self.julian_day += (self.dt / self.secs_per_day) # [days]

        # ------------------------------------------
        # Compute the offset from True Solar Noon
        # clock_hour is in 24-hour military time
        # but it can have a decimal part.
        # ------------------------------------------
        dec_part = self.julian_day - int(self.julian_day)
        clock_hour = dec_part * self.hours_per_day
        ## print '    Computing solar_noon...'
        self.GMT_offset = solar.gmt_offset_hours(lat=self.cfg.lat, lon=self.cfg.lon, when_utc=self.start_datetime)
        solar_noon = solar.True_Solar_Noon(
            self.julian_day,
            self.cfg.lon,    # for USA region, lon is negative
            self.GMT_offset,    #  time-zone offset from GMT/UTC in hours
            DST_offset=None,  #####
            year=self.year,
        )
        ## print '    Computing TSN_offset...'
        self.TSN_offset = clock_hour - solar_noon  # [hours]

    def update_albedo(self, method="aging"):
        """Only use this routine if time varying albedo is not supplied as an input:
        ------------------------------------------------
        Dynamic albedo accounting for aging snow
        ------------------------------------------------
        (Rohrer and Braun 1994): alpha = alpha0 + K * e^(-nr)
        alpha = albedo
        alpha0 = minimum snowpack albedo (~0.4)
        K = constant (~0.44)
        n = number of days since last major snowfall, at least 3 cm over 3 days
        r = recession coefficient = 0.05 for temperatures < than
        0 deg C, 0.12 for temperatures > 0 deg C
        ------------------------------------------------
        """  # noqa: D205
        if method == "aging":
            albedo = self.albedo

            r = np.where((self.T_air > 0), 0.12, 0.05)
            K = 0.44
            alpha0 = 0.4

            self.P_snow_3day_watershed = np.roll(
                self.P_snow_3day_watershed, -1, axis=0
            )  # you can roll on different axes (time axis), shape of the DEM and time axis and roll on the time axis
            ws_density_ratio = self.rho_H2O / self.rho_snow
            self.P_snow_3day_watershed[np.size(self.P_snow_3day_watershed, axis=0) - 1] = (
                self.P_snow * self.dt * ws_density_ratio
            )

            p_snow_3day_watershed_total = np.sum(
                self.P_snow_3day_watershed, axis=0
            )  # maybe multipy by timestep here # also make sure to only sum over time axis
            # self.p_snow_3day_watershed_total = p_snow_3day_watershed_total # if you want to output and make sure it's working properly

            self.n = np.where((p_snow_3day_watershed_total >= 0.03), 0, self.n)
            self.n = np.where((p_snow_3day_watershed_total < 0.03), self.n + self.days_per_dt, self.n)
            snow_albedo = alpha0 + K * np.exp(-self.n * r)

            albedo = np.where(
                (self.h_snow > 0),  # where snow exists
                snow_albedo,
                albedo,
            )
            albedo = np.where(
                ((self.h_snow == 0) & (self.h_ice > 0)),  # where ice exists without snow
                np.float64(0.3),
                albedo,
            )
            albedo = np.where(
                ((self.h_snow == 0) & (self.h_ice == 0)),  # where there is no snow or ice (tundra)
                np.float64(0.15),
                albedo,
            )
            self.albedo = albedo
        # ------------------------------------------------
        # Simple dynamic albedo depending on ice vs. snow vs. bare ground (tundra) using values from Dingman
        # ------------------------------------------------
        if method == "simple":
            albedo = self.albedo
            albedo = np.where(
                (self.h_snow > 0),  # where snow exists
                np.float64(0.75),
                albedo,
            )
            albedo = np.where(
                ((self.h_snow == 0) & (self.h_ice > 0)),  # where ice exists without snow
                np.float64(0.3),
                albedo,
            )
            albedo = np.where(
                ((self.h_snow == 0) & (self.h_ice == 0)),  # where there is no snow or ice (tundra)
                np.float64(0.15),
                albedo,
            )
            self.albedo = albedo

    def set_aspect_angle(self):
        # ------------------------------------------------------
        # ---------------------------------------------------------
        alpha = (np.pi / 2) - self.cfg.aspect
        alpha = (self.twopi + alpha) % self.twopi
        # -----------------------------------------------
        is_nan = not np.isfinite(alpha)
        if is_nan:
            alpha = np.float64(0)

        self.alpha = alpha

    def set_slope_angle(self):

        # -------------------------------------------------
        # -------------------------------------------------
        self.slopes = self.cfg.slope
        beta = np.arctan(self.cfg.slope)
        beta = (self.twopi + beta) % self.twopi
        # ---------------------------------------------
        is_nan = not np.isfinite(beta)
        if is_nan:
            beta = np.float64(0)
        # ------------------------------------------------------------------
        w_bad = np.logical_or((beta < 0), (beta > np.pi / 2))
        if w_bad:
            print('ERROR: In met_base.py, some slope angles are out')
            print('       of range.  Returning without setting beta.')
            print()
            return

        self.beta = beta

    def update_net_shortwave_radiation(self):
        """Notes:  If time is before local sunrise or after local
        #         sunset then Qn_SW should be zero.
        # ---------------------------------------------------------
        # Compute Qn_SW for this time
        # --------------------------------
        """  # noqa: D205
        K_cs = solar.Clear_Sky_Radiation(
            self.cfg.lat,
            self.julian_day,
            self.W_p,
            self.TSN_offset,
            self.alpha,
            self.beta,
            self.albedo,
            self.cfg.dust_atten,
        )

        # -------------------------------------------
        # 2024-03-06: Fix missing account for albedo
        # in net shortwave radiation calcs
        # Dingman 3rd Edition 2015 Eqn. 6B1.1:
        # net shortwave = Kin * (1-albedo)
        # -------------------------------------------
        Qn_SW = K_cs * (1 - self.albedo)

        if np.ndim(self.Qn_SW) == 0:
            self.Qn_SW.fill(Qn_SW)  #### (mutable scalar)
        else:
            self.Qn_SW[:] = Qn_SW  # [W m-2]

    def update_em_air(self):
        """NB!  The Brutsaert and Satterlund formulas for air
        emissivity as a function of air temperature are in
             close agreement; see compare_em_air_methods().
             However, we must pay close attention to whether
             equations require units of kPa, Pa, or mbar.

                    100 kPa = 1 bar = 1000 mbars
                       => 1 kPa = 10 mbars
        ---------------------------------------------------------
        NB!  Temperatures are assumed to be given with units
             of degrees Celsius and are converted to Kelvin
             wherever necessary by adding C_to_K = 273.15.

             RH = relative humidity [unitless]
        ---------------------------------------------------------
        NB!  I'm not sure about how F is added at end because
             of how the equation is printed in Dingman (2002).
             But it reduces to other formulas as it should.
        ---------------------------------------------------------
        """  # noqa: D205
        T_air_K = self.T_air + self.C_to_K

        if not (self.cfg.SATTERLUND):
            # -----------------------------------------------------
            # Brutsaert (1975) method for computing emissivity
            # of the air, em_air.  This formula uses e_air with
            # units of kPa. (From Dingman (2002, p. 196).)
            # See notes for update_vapor_pressure().
            # -----------------------------------------------------
            e_air_kPa = self.e_air / np.float64(10)  # [kPa]
            F = self.cfg.canopy_factor
            C = self.cfg.cloud_factor
            term1 = (1.0 - F) * 1.72 * (e_air_kPa / T_air_K) ** self.one_seventh
            term2 = 1.0 + (0.22 * C**2.0)
            em_air = (term1 * term2) + F
        else:
            # --------------------------------------------------------
            # Satterlund (1979) method for computing the emissivity
            # of the air, em_air, that is intended to "correct
            # apparent deficiencies in this formulation at air
            # temperatures below 0 degrees C" (see G. Liston)
            # Liston cites Aase and Idso(1978), Satterlund (1979)
            # --------------------------------------------------------
            e_air_mbar = self.e_air
            eterm = np.exp(-1 * (e_air_mbar) ** (T_air_K / 2016))
            em_air = 1.08 * (1.0 - eterm)

        # --------------------------
        # Update em_air, in-place
        # ---------------------------------------------------------
        # NB! Currently, em_air is always initialized as scalar,
        #     but could change to grid after assignment.  Must
        #     determine if scalar or grid in initialize().
        # ---------------------------------------------------------
        self.em_air = em_air
        #         if (np.ndim( self.em_air ) == 0):
        #             self.em_air.fill( em_air )   #### (mutable scalar)
        #         else:
        #             self.em_air[:] = em_air

    def update_net_longwave_radiation(self):
        """Notes: Net longwave radiation is computed using the
               Stefan-Boltzman law.  All four data types
               should be allowed (scalar, time series, grid or
               grid stack).

               Qn_LW = (LW_in - LW_out)
               LW_in   = em_air  * sigma * (T_air  + 273.15)^4
               LW_out  = em_surf * sigma * (T_surf + 273.15)^4

               Temperatures in [deg_C] must be converted to
               [K].  Recall that absolute zero occurs at
               0 [deg_K] or -273.15 [deg_C].

        ----------------------------------------------------------------
        First, e_air is computed as:
          e_air = RH * 0.611 * exp[(17.3 * T_air) / (T_air + 237.3)]
        Then, em_air is computed as:
          em_air = (1 - F) * 1.72 * [e_air / (T_air + 273.15)]^(1/7) *
                    (1 + 0.22 * C^2) + F
        ----------------------------------------------------------------
        Compute Qn_LW for this time
        --------------------------------
        """  # noqa: D205

        T_surf_K = self.T_surf + self.C_to_K
        # LW_in is alread available from inputs
        # T_air_K = self.T_air + self.C_to_K
        # LW_in = self.em_air * self.cfg.sigma * (T_air_K) ** 4.0
        LW_in = self.LW_in
        LW_out = self.cfg.em_surf * self.cfg.sigma * (T_surf_K) ** 4.0

        # ----------------------------------------------------
        # 2023-08-29.  The next line was here before today,
        # and accounts for the amount of longwave radiation
        # from the air that is reflected from the surface.
        # See: https://daac.ornl.gov/FIFE/guides/
        #        Longwave_Radiation_UNL.html
        # It reduces the net longwave radiation.
        # ----------------------------------------------------
        LW_out += (1.0 - self.cfg.em_surf) * LW_in

        self.Qn_LW = LW_in - LW_out  # [W m-2]

        # --------------------------------------------------------------
        # Can't do this yet.  Qn_LW is always initialized grid now
        # but will often be created above as a scalar. (9/23/14)
        # --------------------------------------------------------------
        #         if (np.ndim( self.Qn_LW ) == 0):
        #             self.Qn_LW.fill( Qn_LW )   #### (mutable scalar)
        #         else:
        #             self.Qn_LW[:] = Qn_LW  # [W m-2]

    def update_net_energy_flux(self):
        """Notes: Q_sum is used by "snow_energy_balance.py".
        ------------------------------------------------------
               Qm    = energy used to melt snowpack (if > 0)
               Qn_SW = net shortwave radiation flux (solar)
               Qn_LW = net longwave radiation flux (air, surface)
               Qh    = sensible heat flux from turbulent convection
                       between snow surface and air
               Qe    = latent heat flux from evaporation, sublimation,
                       and condensation
               Qa    = energy advected by moving water (i.e. rainfall)
                       (ARHYTHM assumes this to be negligible; Qa=0.)
               Qc    = energy flux via conduction from snow to soil
                       (ARHYTHM assumes this to be negligible; Qc=0.)
               Ecc   = cold content of snowpack = amount of energy
                       needed before snow can begin to melt [J m-2]

               All Q's here have units of [W m-2].
               Are they all treated as positive quantities ?

               rho_air  = density of air [kg m-3]
               rho_snow = density of snow [kg m-3]
               Cp_air   = specific heat of air [J kg-1 K-1]
               Cp_snow  = heat capacity of snow [J kg-1 K-1]
                        = ???????? = specific heat of snow
               Kh       = eddy diffusivity for heat [m2 s-1]
               Ke       = eddy diffusivity for water vapor [m2 s-1]
               Lv       = latent heat of vaporization [J kg-1]
               Lf       = latent heat of fusion [J kg-1]
               ------------------------------------------------------
               Dn       = bulk exchange coeff for the conditions of
                          neutral atmospheric stability [m/s]
               Dh       = bulk exchange coeff for heat
               De       = bulk exchange coeff for vapor
               ------------------------------------------------------
               T_air    = air temperature [deg_C]
               T_surf   = surface temperature [deg_C]
               T_snow   = average snow temperature [deg_C]
               RH       = relative humidity [unitless] (in [0,1])
               e_air    = air vapor pressure at height z [mbar]
               e_surf   = surface vapor pressure [mbar]
               ------------------------------------------------------
               h_snow   = snow depth [m]
               z        = height where wind speed is uz [m]
               uz       = wind speed at height z [m/s]
               p0       = atmospheric pressure [mbar]
               T0       = snow temperature when isothermal [deg_C]
                          (This is usually 0.)
               z0_air   = surface roughness length scale [m]
                          (includes vegetation not covered by snow)
                          (Values from page 1033: 0.0013, 0.02 [m])
               kappa    = von Karman's constant [unitless] = 0.41
               dt       = snowmelt timestep [seconds]
        ----------------------------------------------------------------
        """  # noqa: D205
        Q_sum = self.Qn_SW + self.Qn_LW + self.Qh + self.Qe + self.Qa + self.Qc  # [W m-2]

        if np.ndim(self.Q_sum) == 0:
            self.Q_sum.fill(Q_sum)  #### (mutable scalar)
        else:
            self.Q_sum = Q_sum  # [W m-2]

    def update_snow_meltrate(self):
        """Compute energy-balance meltrate
        # ------------------------------------------------------
        # Eccs is initialized by initialize_snow_cold_content().
        # ------------------------------------------------------
        # The following pseudocode only works for scalars but
        # is otherwise equivalent to that given below and
        # clarifies the logic:
        # ------------------------------------------------------
        #  if (Q_sum gt 0) then begin
        #      if ((Q_sum * dt) gt Eccs) then begin
        #          ;-------------------------------------------
        #          ; Snow is melting.  Use some of Q_sum to
        #          ; overcome Eccs, and remainder to melt snow
        #          ;-------------------------------------------
        #          Qm  = Q_sum - (Eccs/dt)
        #          Eccs = 0
        #          M   = (Qm / (rho_w * Lf))
        #      endif else begin
        #          ;------------------------------
        #          ; Snow is warming; reduce Eccs
        #          ;------------------------------
        #          Eccs = (Eccs - (Q_sum * dt))
        #          M   = 0d
        #      endelse
        #  endif else begin
        #      ;--------------------------------
        #      ; Snow is cooling; increase Eccs
        #      ;--------------------------------
        #      Eccs = Eccs - (Q_sum * dt)
        #      M   = 0d
        #  endelse
        # ---------------------------------------------------------
        # Q_sum = Qn_SW + Qn_LW + Qh + Qe + Qa + Qc    # [W m-2]
        # ---------------------------------------------------------

        # -----------------------------------------------
        # New approach; easier to understand
        # -----------------------------------------------
        # E_in  = energy input over one time step
        # E_rem = energy remaining in excess of Eccs
        # -----------------------------------------------
        """  # noqa: D205
        E_in = self.Q_sum * self.dt
        E_rem = np.maximum(E_in - self.Eccs, np.float64(0))
        Qm = E_rem / self.dt  # [W m-2]

        M = Qm / (self.rho_H2O * self.Lf)  # [m/s]
        if np.size(self.SM) == 1:
            M = np.float64(M)  # avoid type change
            self.SM.fill(M)
        else:
            self.SM = M

    def update_ice_meltrate(self):
        """Compute energy-balance meltrate
        # ------------------------------------------------------
        # Ecci is initialized by initialize_ice_cold_content().
        # ------------------------------------------------------
        # The following pseudocode only works for scalars but
        # is otherwise equivalent to that given below and
        # clarifies the logic:
        # ------------------------------------------------------
        #  if (Q_sum gt 0) then begin
        #      if ((Q_sum * dt) gt Ecci) then begin
        #          ;-------------------------------------------
        #          ; Ice is melting.  Use some of Q_sum to
        #          ; overcome Ecci, and remainder to melt ice
        #          ;-------------------------------------------
        #          Qm  = Q_sum - (Ecci/dt)
        #          Ecci = 0
        #          M   = (Qm / (rho_w * Lf))
        #      endif else begin
        #          ;------------------------------
        #          ; Ice is warming; reduce Ecci
        #          ;------------------------------
        #          Ecci = (Ecci - (Q_sum * dt))
        #          M   = 0d
        #      endelse
        #  endif else begin
        #      ;--------------------------------
        #      ; Ice is cooling; increase Ecci
        #      ;--------------------------------
        #      Ecci = Ecci - (Q_sum * dt)
        #      M   = 0d
        #  endelse
        # ---------------------------------------------------------
        # Q_sum = Qn_SW + Qn_LW + Qh + Qe + Qa + Qc    # [W m-2]
        # ---------------------------------------------------------

        # -----------------------------------------------
        # New approach; easier to understand
        # -----------------------------------------------
        # E_in  = energy input over one time step
        # E_rem = energy remaining in excess of Ecci
        # -----------------------------------------------
        """  # noqa: D205
        E_in = self.Q_sum * self.dt
        E_rem = np.maximum(E_in - self.Ecci, np.float64(0))
        Qm = E_rem / self.dt  # [W m-2]

        M = Qm / (self.rho_H2O * self.Lf)  # [m/s]
        IM = np.maximum(M, np.float64(0))
        self.IM = np.where((self.h_swe == 0) & (self.previous_swe == 0), IM, np.float64(0))

        Ecci = np.maximum((self.Ecci - E_in), np.float64(0))

        Ecci = np.where((self.h_ice == 0), np.float64(0), Ecci)

        if np.size(self.Ecci) == 1:
            Ecci = np.float64(Ecci)  # avoid type change
            self.Ecci.fill(Ecci)
        else:
            self.Ecci[:] = Ecci

    def update_combined_meltrate(self):
        """We want to feed combined snow and ice melt to GIUH for
        # runoff, so combine the IM and SM variables to create Mtotal.
        # ---------------------------------------------------------
        """  # noqa: D205
        M_total = self.IM + self.SM

        self.M_total = M_total

    def enforce_max_snow_meltrate(self):
        """The max possible meltrate would be if all snow (given
        # by snow depth, h_snow, were to melt in the one time
        # step, dt.  Meltrate should never exceed this value.
        # Recall that: (h_snow / h_swe) = (rho_H2O / rho_snow)
        #                               = density_ratio > 1
        # So h_swe = h_snow / density_ratio.
        # Previous version had a bug; see below.
        # Now also using "out" keyword for "in-place".
        # -------------------------------------------------------
        SM_max = self.h_swe / self.dt
        self.SM = np.minimum(self.SM, SM_max, out=self.SM)  # [m s-1]

        # ------------------------------------------------------
        # Make sure meltrate is positive, while we're at it ?
        # Is already done by "Energy-Balance" component.
        # ------------------------------------------------------
        """  # noqa: D205
        self.SM = np.maximum(self.SM, np.float64(0))

    def enforce_max_ice_meltrate(self):
        """The max possible meltrate would be if all ice (given
        # by ice depth, h_ice, were to melt in the one time
        # step, dt.  Meltrate should never exceed this value.
        # -------------------------------------------------------
        """  # noqa: D205
        IM_max = self.h_iwe / self.dt
        self.IM = np.minimum(self.IM, IM_max, out=self.IM)  # [m s-1]

        # ------------------------------------------------------
        # Make sure meltrate is positive, while we're at it ?
        # Is already done by "Energy-Balance" component.
        # ------------------------------------------------------
        np.maximum(self.IM, np.float64(0), out=self.IM)

    def update_SM_integral(self):
        """Update mass total for SM, sum over all pixels
        # ------------------------------------------------
        """  # noqa: D205
        volume = np.float64(self.SM * self.cfg.da * self.dt)  # [m^3]
        self.vol_SM += np.sum(volume)  #### np.sum vs. sum ???

    def update_IM_integral(self):
        """Update mass total for IM, sum over all pixels
        # ------------------------------------------------
        """  # noqa: D205
        volume = np.float64(self.IM * self.cfg.da * self.dt)
        self.vol_IM += np.sum(volume)

    def update_snowfall_cold_content(self):
        """Copy previous timestep's CC and adjust from here
        ----------------------------------------------------
        For newly fallen snow, add cold content using
        the same equation used for initializing cold content,
        but use wet bulb temperature as T_snow for new snow:
        --------------------------------------------------------------------
        Wet bulb temp. equation from Stull 2011. Adapted from R code:
        https://github.com/SnowHydrology/humidity/blob/master/R/humidity.R
        --------------------------------------------------------------------
        """  # noqa: D205
        new_h_snow = (self.P_snow * self.dt) * self.ws_density_ratio
        Eccs = self.Eccs

        # ----------------------------------------------------
        # Prepare to adjust CC for land surface energy fluxes
        # ----------------------------------------------------
        E_in = self.Q_sum * self.dt  # [J m-2]
        T_wb = (
            self.T_air * np.arctan(0.151977 * ((self.RH + 8.313659) ** 0.5))
            + np.arctan(self.T_air + self.RH)
            - np.arctan(self.RH - 1.676331)
            + ((0.00391838 * (self.RH**1.5)) * np.arctan(0.023101 * self.RH))
            - 4.86035
        )

        del_T = self.T0_cc - T_wb

        # ----------------------------------------------------
        # Only where NEW snow has fallen (P_snow > 0), ADD
        # cold content for the new snow AND account for land
        # surface energy fluxes
        # ----------------------------------------------------
        Eccs = np.where(
            (self.P_snow > 0),
            (
                np.maximum(
                    (Eccs + ((self.rho_snow * self.Cp_snow) * new_h_snow * del_T) - E_in), np.float64(0)
                )
            ),
            Eccs,
        )  # make sure signs check out
        # Eccs = np.maximum(Eccs, np.float64(0)) # make sure signs check out

        if np.size(self.Eccs) == 1:
            Eccs = np.float64(Eccs)  # avoid type change
            self.Eccs.fill(Eccs)
        else:
            self.Eccs[:] = Eccs

    def update_snowpack_cold_content(self):
        """Copy the CC that was only adjusted in places WITH
        new snowfall before adjusting in places WITHOUT new
        snowfall
        -----------------------------------------------------
        """  # noqa: D205
        Eccs = self.Eccs

        E_in = self.Q_sum * self.dt  # [J m-2]
        # Eccs  = np.maximum((self.Eccs - E_in), np.float64(0))
        Eccs = np.where((self.P_snow <= 0), np.maximum((Eccs - E_in), np.float64(0)), Eccs)

        Eccs = np.where((self.h_snow == 0), np.float64(0), Eccs)

        if np.size(self.Eccs) == 1:
            Eccs = np.float64(Eccs)  # avoid type change
            self.Eccs.fill(Eccs)
        else:
            self.Eccs[:] = Eccs

    def extract_previous_swe(self):
        """Extract swe from previous timestep for use in
        toggling between ice/snow routines
        ------------------------------------------------
        """  # noqa: D205
        self.previous_swe = self.h_swe.copy()

    def update_swe(self):
        """The Meteorology component uses air temperature
        to compute P_rain (precip that falls as liquid) and
        P_snow (precip that falls as snow or ice) separately.
        P_snow = (self.P * (self.T_air <= 0))
        ----------------------------------------------------------
        Note: This method must be written to work regardless
        of whether P_rain and T are scalars or grids. (3/14/07)
        ------------------------------------------------------------
        If P or T_air is a grid, then h_swe and h_snow are grids.
        This is set up in initialize_computed_vars().
        ------------------------------------------------------------

        ------------------------------------------------
        Increase snow water equivalent due to snowfall
        ------------------------------------------------
        Meteorology and Channel components may have
        different time steps, but then self.P_snow
        will be a time-interpolated value.
        ------------------------------------------------
        """  # noqa: D205
        dh1_swe = self.P_snow * self.dt
        self.h_swe += dh1_swe

        # ------------------------------------------------
        # Decrease snow water equivalent due to melting
        # Note that SM depends partly on h_snow.
        # ------------------------------------------------
        dh2_swe = self.SM * self.dt
        self.h_swe -= dh2_swe
        np.maximum(self.h_swe, np.float64(0), self.h_swe)  # (in place)

    def update_iwe(self):
        """Decrease ice water equivalent due to melting
        ------------------------------------------------
        """  # noqa: D205
        dh2_iwe = self.IM * self.dt
        self.h_iwe -= dh2_iwe
        np.maximum(self.h_iwe, np.float64(0), self.h_iwe)  # (in place)

    def update_density_ratio(self):
        """Return if density_ratio is constant in time.
        -----------------------------------------------
        """  # noqa: D205

        density_ratio = self.cfg.rho_H2O / self.cfg.rho_snow

        # -------------------------------------
        # Save updated density ratio in self
        # -------------------------------------
        if np.ndim(self.density_ratio) == 0:
            density_ratio = np.float64(density_ratio)  ### (from 0D array to scalar)
            self.density_ratio.fill(density_ratio)  ### (mutable scalar)
        else:
            self.density_ratio[:] = density_ratio

    def update_swe_integral(self):
        """Update mass total for water in the snowpack,
        sum over all pixels.
        ------------------------------------------------
        """  # noqa: D205
        volume = np.float64(self.h_swe * self.cfg.da)  # [m^3]
        if np.size(volume) == 1:
            self.vol_swe += volume * self.rti.n_pixels
        else:
            self.vol_swe += np.sum(volume)

    def update_iwe_integral(self):
        """Update mass total for water in the ice column,
        sum over all pixels.
        ------------------------------------------------
        """  # noqa: D205
        volume = np.float64(self.h_iwe * self.cfg.da)  # [m^3]
        if np.size(volume) == 1:
            self.vol_iwe += volume * self.rti.n_pixels
        else:
            self.vol_iwe += np.sum(volume)

    def extract_previous_snow_depth(self):
        """Extract swe from previous timestep for use in
        toggling between ice/snow routines
        ------------------------------------------------
        """  # noqa: D205
        self.previous_h_snow = self.h_snow.copy()

    def update_snow_depth(self):
        """The Meteorology component uses air temperature
        to compute P_rain (precip that falls as liquid) and
        P_snow (precip that falls as snow or ice) separately.
        P_snow = (self.P * (self.T_air <= 0))
        ----------------------------------------------------------
        Note: This method must be written to work regardless
        of whether P_rain and T are scalars or grids.
        ------------------------------------------------------------
        If P or T_air is a grid, then h_swe and h_snow are grids.
        This is set up in initialize_computed_vars().
        ------------------------------------------------------------
        Note that for a region of area, A:
            rho_snow = (mass_snow / (h_snow * A))
            rho_H2O  = (mass_H20  / (h_swe * A))
        Since mass_snow = mass_H20 (for SWE):
            rho_snow * h_snow = rho_H2O * h_swe
            (h_snow / h_swe)  = (rho_H2O / rho_snow)
             h_snow = h_swe * density_ratio
        Since liquid water is denser than snow:
             density_ratio > 1 and
             h_snow > h_swe
        self.density_ratio = (self.rho_H2O / self.rho_snow)
        rho_H2O is for liquid water close to 0 degrees C.
        ------------------------------------------------------------

        -------------------------------------------------
        Change snow depth due to melting or falling snow
        -------------------------------------------------
        This assumes that update_swe() is called
        before update_snow_depth().
        -------------------------------------------
        """  # noqa: D205
        h_snow = self.h_swe * self.ws_density_ratio
        if np.ndim(self.h_snow) == 0:
            h_snow = np.float64(h_snow)  ### (from 0D array to scalar)
            self.h_snow.fill(h_snow)  ### (mutable scalar)
        else:
            self.h_snow[:] = h_snow

    def update_ice_depth(self):
        """Change ice depth due to melting
        ---------------------------------
        This assumes that update_iwe() is called
        before update_ice_depth().
        -------------------------------------------
        """  # noqa: D205
        h_ice = self.h_iwe * self.wi_density_ratio
        if np.ndim(self.h_ice) == 0:
            h_ice = np.float64(h_ice)
            self.h_ice.fill(h_ice)
        else:
            self.h_ice[:] = h_ice

    def update_total_snowpack_water_volume(self):
        """Compute the total volume of water stored
               in the snowpack for all grid cells in the DEM.
               Use this in the final mass balance reporting.
               (2023-08-31)
        --------------------------------------------------------
        Note:  This is called from initialize() & finalize().
        --------------------------------------------------------

        ----------------------------------------------------
        Update total volume of liquid water stored in the
        current snowpack, sum over all grid cells but no
        integral over time.  (2023-08-31)
        ----------------------------------------------------
        """  # noqa: D205
        volume = np.float64(self.h_swe * self.da)  # [m^3]
        if np.size(volume) == 1:
            vol_swe = volume * self.rti.n_pixels
        else:
            ## volume[ self.edge_IDs ] = 0.0  # (not needed)
            vol_swe = np.sum(volume)

        self.vol_swe.fill(vol_swe)

    #   update_total_snowpack_water_volume()
    # -------------------------------------------------------------------
    def update_total_ice_water_volume(self):
        """Compute the total volume of water stored
               in the ice for all grid cells in the DEM.
               Use this in the final mass balance reporting.
               (2023-08-31)
        --------------------------------------------------------
        Note:  This is called from initialize() & finalize().
        --------------------------------------------------------

        ----------------------------------------------------
        Update total volume of liquid water stored in the
        current ice, sum over all grid cells but no
        integral over time.  (2023-08-31)
        ----------------------------------------------------
        """  # noqa: D205
        volume = np.float64(self.h_iwe * self.cfg.da)  # [m^3]
        vol_iwe = np.sum(volume)

        self.vol_iwe.fill(vol_iwe)

    def get_component_name(self) -> str:
        """Name of this BMI module component.

        Returns
        -------
            str: Model Name
        """
        return "Topoflow-Glacier"

    def get_input_item_count(self) -> int:
        """Number of model input variables

        Returns
        -------
            int: number of input variables
        """
        return len(input_names)

    def get_input_var_names(self) -> tuple[str, ...]:  # type: ignore[override]
        """The names of each input variables

        Returns
        -------
            tuple[str, ...]: iterable tuple of input variable names
        """
        return self.input_names

    def get_output_item_count(self) -> int:
        """Number of model output variables

        Returns
        -------
            int: number of output variables
        """
        return len(self.output_names)

    def get_output_var_names(self) -> tuple[str, ...]:  # type: ignore[override]
        """The names of each output variable

        Returns
        -------
            tuple[str, ...]: iterable tuple of output variable names
        """
        return self.output_names

    # @FRahmani368 Please work on this some more to map forcings into the variables
    def set_value(self, name: str, src: Any) -> None:
        """Sets and input or output value

        Args:
            name (str): name of value
            src (Any): value to set

        Raises
        ------
            ValueError: If name does not exist
        """
        ## outputs variables mapping
        if name in _var_name_map:
            self._output[name] = src
        elif name in self._dynamic_inputs.names():
            self._dynamic_inputs.set_value(name, src)
            # dynamic input vaiable mapping
            if name in _input_alias_map:
                attr = _input_alias_map.get(name)
                # Create or update the attribute (array or scalar is fine)
                setattr(self, attr, src)
        else:
            raise ValueError(
                f"Variable {name} does not exist input or output variables.  User getters to view options."
            )

    def get_value(self, name: str, dest: NDArray) -> NDArray:
        """_Copies_ a variable's np.np.ndarray into `dest` and returns `dest`."""
        value = self.get_value_ptr(name)
        try:
            if not isinstance(value, np.ndarray):
                dest[:] = np.array(value).flatten()
            else:
                dest[:] = self.get_value_ptr(name).flatten()
        except Exception as e:
            raise RuntimeError(f"Could not return value {name} as flattened array") from e

        return dest

    def get_value_ptr(self, name: str) -> NDArray:
        """Gets value in native form if exists in inputs or outputs"""
        try:
            return self.output[name]
        except KeyError:
            return self.input[name]
        except KeyError as e:  # NOQA: B025
            raise KeyError(f"{name} is not a known variable") from e

    def get_var_itemsize(self, name: str) -> int:
        """Size, in bytes, of a single element of the variable name

        Args:
            name (str): variable name

        Returns
        -------
            int: number of bytes representing a single variable of @p name
        """
        return self.get_value_ptr(name).itemsize

    def get_var_nbytes(self, name: str) -> int:
        """Size, in nbytes, of a single element of the variable name

        Args:
            name (str): Name of variable.

        Returns
        -------
            int: Size of data array in bytes.
        """
        return self.get_value_ptr(name).nbytes

    def get_var_type(self, name: str) -> str:
        """Data type of variable.

        Args:
            name (str): Name of variable.

        Returns
        -------
            str: Data type.
        """
        return str(self.get_value_ptr(name).dtype)

    def get_current_datetime(self, time_units="seconds"):
        """
        Advance start_datetime by a given offset.
        Returns a pandas.Timestamp.

        Parameters
        ----------
        start_datetime : pd.Timestamp | str | datetime
        time : int | float
            Amount to advance. Can be fractional for seconds/minutes/hours/days.
        time_units : {"seconds","minutes","hours","days"}
        """

        time = self.cfg.dt

        if not isinstance(self.start_datetime, pd.Timestamp):
            self.start_datetime = pd.to_datetime(self.start_datetime)

        if time_units in ("second", "seconds", "s", "sec"):
            self.start_datetime += pd.to_timedelta(time, unit="s")
        elif time_units in ("minute", "minutes", "min"):
            self.start_datetime += pd.to_timedelta(time, unit="m")
        elif time_units in ("hour", "hours", "hr", "hrs"):
            self.start_datetime += pd.to_timedelta(time, unit="h")
        elif time_units in ("day", "days", "d"):
            self.start_datetime += pd.to_timedelta(time, unit="d")
        else:
            raise ValueError(f"Unsupported time_units: {time_units}")