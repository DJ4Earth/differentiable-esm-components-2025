using Enzyme, Checkpointing, ShallowWaters
using NetCDF, JLD2, CairoMakie, HDF5
using LinearAlgebra, Random
using NLPModels, MadNLP

Enzyme.API.looseTypeAnalysis!(true)
Random.seed!(52)

include("energy_loss.jl")
include("da_initialcond.jl")