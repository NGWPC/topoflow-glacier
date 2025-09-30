# Topoflow-Glacier

Topoflow-Glacier was created from the extraction of the Glacier Module from [Topoflow](https://github.com/NOAA-OWP/topoflow/tree/hydrofabric_fixes) in order to be integrated into the NextGen Framework under the Next Generation Water Prediction Capability. The following docs detail the specific functions of Topoflow-Glacier, and some test code at a domain of interest.

## Getting Started

This repo is managed through [UV](https://docs.astral.sh/uv/getting-started/installation/) and can be installed through:
```sh
uv sync
```

Once the venv is installed, examples can be run through
```sh
uv run python examples/run_topoflow_glacier.py
```

## Tests
To run all tests, run
```sh
uv sync --all-extras
uv run pytest
```
