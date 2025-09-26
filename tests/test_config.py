from pathlib import Path

import pandas as pd
import yaml


def test_mock_config_and_forcing_file(mock_config: Path) -> None:
    """Test that the mock config YAML can be parsed and the forcing file exists and is readable"""

    # Test 1: Parse the YAML configuration
    with open(mock_config) as f:
        cfg = yaml.safe_load(f)

    # Test 2: Verify all expected keys are present in the config
    expected_keys = [
        "site_prefix",
        "forcing file",
        "n_steps",
        "dt",
        "Cp_snow",
        "Cp_ice",
        "rho_snow",
        "rho_ice",
        "h_active_layer",
        "T0",
        "h0_snow",
        "h0_ice",
        "h0_swe",
        "h0_iwe",
    ]

    for key in expected_keys:
        assert key in cfg, f"Missing key '{key}' in config"

    # Test 3: Verify forcing file path exists
    forcing_file_path = Path(cfg["forcing file"])
    assert forcing_file_path.exists(), f"Forcing file does not exist: {forcing_file_path}"

    # Test 5: Read and validate the forcing file CSV
    df = pd.read_csv(forcing_file_path)

    # Test 7: Verify the forcing file has data
    assert len(df) > 0, "Forcing file is empty"
