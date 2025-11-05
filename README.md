# technicalpaper-2025
Code to reproduce experiments and figures in JAMES 2025 technical paper.

## For section 3 (Application 1: Shallow Water Model in a Rotating System)

See the subdirectory `shallow-water-model`. To run code first include the necessary packages and scripts.
```
include("technical_paper.jl)
```
The two experiments, ensitivity analysis and data assimilation, are in `energy_loss.jl` and `da_initcond.jl` respectively. To run the energy loss
```
Ndays = 10
model, dmodel = run_energy_loss(Ndays)
```
and to run the data assimilation problem,
```
Ndays = 10
result = compute_initcond_newoptimizer(Ndays)
```

## For section 4 (Application 2: Ocean General Circulation Model in a Re-entrant Channel Configuration)

See the linked submodule `re-entrant-channel-model`. Instructions on how to run and replicate results from the manuscript are in the README there.
