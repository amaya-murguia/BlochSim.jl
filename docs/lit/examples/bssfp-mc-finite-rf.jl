#=
# [bSSFP multi-pool with finite RF pulse effects](@id bssfp-mc-finite-rf)

This page extends the bSSFP multi-pool demo to simulate and fit data
using a 2-pool bSSFP signal model taking into account finite RF pulse effects
=#

### amm check: add citations

#=
### Setup

First we add the Julia packages that are need for this demo.
Change `false` to `true` in the following code block
if you are using any of the following packages for the first time.
=#

if false
    import Pkg
    Pkg.add([
        "BlochSim"
        "InteractiveUtils"
        "LaTeXStrings"
        "LinearAlgebra"
        "MIRTjim"
        "MRIPulses"
        "Optim"
        "Plots"
        "Random"
        "Measures"
    ])
end

# Tell this Julia session to use the following packages for this example.
# Run `Pkg.add()` in the preceding code block first, if needed.

### amm check: change which functions use for this demo, copied this from the previous one
using ADTypes: AutoForwardDiff
using Optim: optimize # must precede BlochSim for extension
using BlochSim: GAMMA, Gradient, GradientSpoiling, Position, Spin, SpinMC
using BlochSim: InstantaneousRF, RF, b1_gauss, rf_slice, rf_gauss, duration
using BlochSim: excite!, spoil!
using BlochSim: bssfp, real_imag, signal, snr2sigma, fit_signal, crb
using LinearAlgebra: diag, dot, norm
using MIRTjim: prompt
using MRIPulses: dzrf
using Plots: default, gui
using Plots: plot, plot!, scatter!, scatter3d!, text
using Random: seed!
using Measures: mm

default(titlefontsize = 10, markerstrokecolor = :auto, label="", linewidth = 2)
seed!(0);

# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() || prompt(:draw);

# Define some useful helper functions. Some of these functions 
# put the parameters in the correct format for the multi-compartment 
# spin object constructors.
Hz_to_kHz(Δf_Hz) = Δf_Hz * 10^(-3) # convert frequencies in Hz to kHz
kHz_to_Hz(Δf_kHz) = Δf_kHz * 10^(3); # convert frequencies in kHz to Hz

"""
- In: `f_f` fast fraction (myelin fraction)
- Out: `mwf_tuple` tuple with fast and slow fractions
"""
get_mwf_tuple(f_f) = (f_f, 1-f_f)


"""
    get_τ_tuple(τ_fs_ms, f_f)
# In:
- `τ_fs_ms` residence time for exchange from myelin to non-myelin water (ms)
- `f_f` fast fraction (myelin fraction)
# Out:
- `τ_tuple_ms` tuple with fast-to-slow and slow-to-fast residence times
"""
function get_τ_tuple(τ_fs_ms, f_f)
    τ_sf_ms = (1-f_f) * τ_fs_ms / f_f
    return (τ_fs_ms, τ_sf_ms)
end

"""
# In:
- `Δϕ_rad` RF phase cycling value (radians)
- `Δf0_Hz` off-resonance value (Hz)
- `Δf_myelin_Hz` # additional off-resonance value only experienced by myelin water (Hz)
- `TR_ms` repetition time (ms)
# Out:
- `Δf_tuple_Hz` tuple with off-resonance values for fast and slow compartments

[Hinshaw, J. Appl. Phys. 1976](https://doi.org/10.1063/1.323136).
"""
function get_Δf_tuple(Δϕ_rad, Δf0_Hz, Δf_myelin_Hz, TR_ms)

    ## convert the RF phase cycling value to Hz from radians
    Δϕ_Hz = kHz_to_Hz((Δϕ_rad) / (2π*TR_ms))

    ## subtract the RF phase cycling value from the off-resonance value
    Δf_RF_Hz = Δf0_Hz - Δϕ_Hz

    ## add the myelin off-resonance for the myelin term
    Δf_myelin_RF_Hz = Δf_RF_Hz + Δf_myelin_Hz

    ## create and return tuple for the spin object constructor
    Δf_tuple_Hz = (Δf_myelin_RF_Hz, Δf_RF_Hz)
    return Δf_tuple_Hz
end;

#=
Define tissue parameters.
Values taken from [2] and [4]. #amm edited some from previous version
=#

Mz0 = 0.77 # initial condition for longitudinal magnetization (constant)
f_f = 0.15 # myelin water fraction (MWF), fast fraction
τ_fs_ms = 100.0 # mean residence time

T1_f_ms = 400 # T1 for fast-relaxing, myelin water compartment
T1_s_ms = 850 # T1 for slow-relaxing, non-myelin water compartment
T1_ms_tuple = (T1_f_ms, T1_s_ms)

T2_f_ms = 10 # T2 for fast-relaxing, myelin water compartment
T2_s_ms = 80 # T2 for slow-relaxing, non-myelin water compartment
T2_ms_tuple = (T2_f_ms, T2_s_ms)

Δf_myelin_Hz = 5.0 # frequency shift of myelin water
Δf_Hz = 0.0 # set off-resonance to zero for this plot
Δf_Hz_tuple = (Δf_myelin_Hz, Δf_Hz)

TR_ms, TE_ms = 8, 4; # scan parameters
# flip_ang_arr_deg = [10, 40] # flip angles
# Δϕ_arr_deg = [0, 90, 180] # RF phase cycling value (degrees)
α_deg = 10
α_rad = deg2rad(α_deg)

#=
# Slice-selective case
=#

#=
# Array of spins
For a range of z-positions to examine slice profile.
=#

make_mc_spins(Mz0, f_f, τ_fs_ms, T1_f_ms, T1_s_ms, T2_f_ms, T2_s_ms, Δf_myelin_Hz, Δf_Hz, zpos) = map(zpos) do z
    pos = Position(0, 0, z)
    SpinMC(Mz0, get_mwf_tuple(f_f), T1_ms_tuple, T2_ms_tuple, Δf_Hz_tuple, get_τ_tuple(τ_fs_ms,f_f), pos)
end;

#=
# RF pulse for slice-selection
We begin with a simple truncated sinc pulse.
An SLR pulse would have less ripple,
but worse transition bands
that could greatly affect bSSFP signal.
=#

tRF_ms = 1
nlobe = 3
slice_width_cm = 0.1 # 1 mm slice
Δt_ms = 4e-3 # 4μs dwell time for RF sampling
rf1, rephasing1 = rf_slice(tRF_ms ; α_rad, nlobe, slice_width_cm, Δt_ms) # part of blochsim in slice.jl

# amm just copied from the previous one
# Helper function for modeling B1+ scaling `kappa` below
work2 = (; α_rad, rf = deepcopy(rf1), rf_α = deepcopy(rf1.α), rephasing = rephasing1)
function rf_maker2!(kappa, α_rad)
    copyto!(work2.rf.α, work2.rf_α * (kappa * α_rad / work2.α_rad))
    return work2.rf, work2.rephasing
end
@assert rf_maker2!(1, α_rad)[1].α == rf1.α

function rf_maker2(kappa, α_rad)
    rf, rephasing = rf_slice(tRF_ms ; α_rad=kappa*α_rad, nlobe, slice_width, Δt_ms)
    return rf, rephasing
end
@assert rf_maker2(1, α_rad)[1].α == rf1.α

#=
# Excite the spins with the RF, then apply rephasing gradient
=#
function do_excite!(spins, rf, rephasing)
    map(spins) do spin
        excite!(spin, rf)
        spoil!(spin, rephasing)
    end
end;

#=
# Examine slice-profile effect on qMRI with bSSFP
=#

# src the semicolon in front makes this a named tuple
# src parameters to be estimated
kappa = 1 # also estimate the B1+ factor
xt2 = (; Mz0, f_f, τ_fs_ms, T1_f_ms, T1_s_ms, T2_f_ms, T2_s_ms, Δf_myelin_Hz, Δf_Hz, kappa) # tuple
x = collect(Float64, xt) # unknowns in vector

