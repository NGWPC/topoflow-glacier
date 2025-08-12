from topoflow_glacier.glacier_energy_balance import glacier_component

WV_glacier = glacier_component()

## building the glacier energy balance
WV_glacier.update_snow_meltrate()

print("end")
