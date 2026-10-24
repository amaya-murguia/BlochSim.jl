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
using Optim: optimize, LBFGS, Fminbox # must precede BlochSim for extension
using BlochSim: GAMMA, Gradient, GradientSpoiling, Position, Spin, SpinMC
using BlochSim: InstantaneousRF, RF, b1_gauss, rf_slice, rf_gauss, duration
using BlochSim: excite!, spoil!
using BlochSim: bssfp, real_imag, signal, snr2sigma, fit_signal, crb
using LinearAlgebra: diag, dot, norm
using MIRTjim: prompt
using MRIPulses: dzrf
using Plots: default, gui
using Plots: plot, plot!, scatter!, scatter3d!, text, bar, bar!
using Plots: @layout, annotate!, text, Shape
using Plots.PlotMeasures: mm
using Random: seed!
using Measures: mm
using Printf: @sprintf
using Colors: RGB
using JLD: load, @save

default(titlefontsize = 10, markerstrokecolor = :auto, label="", linewidth = 2.5)
seed!(0);
# logocolors = Colors.JULIA_LOGO_COLORS


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

T2_f_ms = 10 # T2 for fast-relaxing, myelin water compartment
T2_s_ms = 80 # T2 for slow-relaxing, non-myelin water compartment

Δf_myelin_Hz = 5.0 # frequency shift of myelin water
Δf_Hz = 0.01 # set off-resonance to zero for this plot

TR_ms, TE_ms = 8, 4; # scan parameters

#=
# Slice-selective case
=#

#=
# Array of spins
For a range of z-positions to examine slice profile.
=#

make_mc_spins(Mz0, f_f, τ_fs_ms, T1_f_ms, T1_s_ms, T2_f_ms, T2_s_ms, Δf_myelin_Hz, Δf_Hz, zpos) = map(zpos) do z
    pos = Position(0, 0, z)
    T1_ms_tuple = (T1_f_ms, T1_s_ms)
    T2_ms_tuple = (T2_f_ms, T2_s_ms)
    Δf_Hz_tuple = (Δf_myelin_Hz, Δf_Hz)
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
# src check what next two lines are for!!
α_deg = 60 # flip angle ° (somewhat large for testing)
α_rad = deg2rad(α_deg)
slice_width_cm = 0.1 # 1 mm slice
Δt_ms = 4e-3 # 4μs dwell time for RF sampling
rf1, rephasing1 = rf_slice(tRF_ms ; α_rad=α_rad, Δt_ms=Δt_ms, nlobe=nlobe, slice_width=slice_width_cm) # part of blochsim in slice.jl

# amm just copied from the previous one
# Helper function for modeling B1+ scaling `kappa` below
work2 = (; α_rad, rf = deepcopy(rf1), rf_α = deepcopy(rf1.α), rephasing = rephasing1)
function rf_maker2!(kappa, α_rad)
    copyto!(work2.rf.α, work2.rf_α * (kappa * α_rad / work2.α_rad))
    return work2.rf, work2.rephasing
end
@assert rf_maker2!(1, α_rad)[1].α == rf1.α
# src: do I need work2 and rf_maker2! ?

function rf_maker2(kappa, α_rad)
    rf, rephasing = rf_slice(tRF_ms ; α_rad=kappa*α_rad, nlobe, slice_width=slice_width_cm, Δt_ms)
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
xt = (; Mz0, f_f, τ_fs_ms, T1_f_ms, T1_s_ms, T2_f_ms, T2_s_ms, Δf_myelin_Hz, Δf_Hz, kappa) # tuple
x = collect(Float64, xt) # unknowns in vector

lower = [0.01, 0.05, 20,  100,  200,  4,   20,  2, -10, 0.9]
upper = [2,    0.4,  500, 700,  1200, 100, 500, 10, 10, 1.1]

τ_fs_ms_1 = 20
xt1 = (; Mz0, f_f, τ_fs_ms_1, T1_f_ms, T1_s_ms, T2_f_ms, T2_s_ms, Δf_myelin_Hz, Δf_Hz, kappa) # tuple
x1 = collect(Float64, xt1)

τ_fs_ms_2 = 100
xt2 = (; Mz0, f_f, τ_fs_ms_2, T1_f_ms, T1_s_ms, T2_f_ms, T2_s_ms, Δf_myelin_Hz, Δf_Hz, kappa) # tuple
x2 = collect(Float64, xt2)

τ_fs_ms_3 = 500
xt3 = (; Mz0, f_f, τ_fs_ms_3, T1_f_ms, T1_s_ms, T2_f_ms, T2_s_ms, Δf_myelin_Hz, Δf_Hz, kappa) # tuple
x3 = collect(Float64, xt3)

f_f_4 = 0.10
xt4 = (; Mz0, f_f_4, τ_fs_ms, T1_f_ms, T1_s_ms, T2_f_ms, T2_s_ms, Δf_myelin_Hz, Δf_Hz, kappa) # tuple
x4 = collect(Float64, xt4)

f_f_5 = 0.20
xt5 = (; Mz0, f_f_5, τ_fs_ms, T1_f_ms, T1_s_ms, T2_f_ms, T2_s_ms, Δf_myelin_Hz, Δf_Hz, kappa) # tuple
x5 = collect(Float64, xt5)

f_f_6 = 0.30
xt6 = (; Mz0, f_f_6, τ_fs_ms, T1_f_ms, T1_s_ms, T2_f_ms, T2_s_ms, Δf_myelin_Hz, Δf_Hz, kappa) # tuple
x6 = collect(Float64, xt6)

#=
Hand-crafted scan "design"
with various phase-cycling increments and flip angles:
=#
Δϕ_rads = (1:8)/8 * 2π .- π # phase-cycling factors
# α_degs = [10, 30, 50] # flip angles
α_degs = [10, 20, 30] # flip angles
α_rads = deg2rad.(α_degs)
design = Iterators.product(Δϕ_rads, α_rads)
num_scans = length(design) # number of different scans

# different design
Δϕ_rads_20 = (1:20)/20 * 2π .- π # phase-cycling factors
α_degs_20 = [10, 20, 30, 40, 50, 60] # flip angles
α_rads_20 = deg2rad.(α_degs_20)
design_20 = Iterators.product(Δϕ_rads_20, α_rads_20)

# Helper for `InstantaneousRF` bSSFP signal model
# src: todo: currently must allocate because InstantaneousRF is immutable
# src: this bssfp version takes in a SpinMC object
function _bssfp0(x, Δϕ_rad, α_rad)
    one_spin = make_mc_spins(x[1:9]..., 0) # this does make 1 spin (maybe should change function name)
    return bssfp(one_spin, TR_ms, TE_ms, Δϕ_rad, InstantaneousRF(x[10] * α_rad));
end


#=
Helper for bSSFP model with slice-selective RF pulse.
- Here we must sum across spins the effect of each `(Δϕ_rad, α_rad)` pair.
- This version models flip-angle dependent slice profile effects.
=#
function _bssfp1(x, Δϕ_rad, α_rad, scale, rf_maker, zpos)
    rf, rephasing = rf_maker(x[10], α_rad) # kappa
    spins = make_mc_spins(x[1:9]..., zpos) # todo: mutable Spin would avoid alloc
    return scale * sum(spins) do spin
         bssfp(spin, TR_ms, TE_ms, Δϕ_rad, (rephasing, rf, rephasing))
    end
end

#=
Signal models
=#

function signal2_c0(x) # signal model for `InstantaneousRF` 2-pool bSSFP signal model
    _bssfp(Δϕ_rad, α_rad) = _bssfp0(x, Δϕ_rad, α_rad) # defines another helper function
    return map(splat(_bssfp), design)
end

function signal2_c0_20(x) # signal model for `InstantaneousRF` 2-pool bSSFP signal model
    _bssfp(Δϕ_rad, α_rad) = _bssfp0(x, Δϕ_rad, α_rad) # defines another helper function
    return map(splat(_bssfp), design_20)
end


function signal2_c1(x, scale, rf_maker, zpos) # 2-pool signal model with slice-selective RF pulse
    _bssfp(Δϕ_rad, α_rad) = _bssfp1(x, Δϕ_rad, α_rad, scale, rf_maker, zpos) # defines another helper function
    return map(splat(_bssfp), design)
end;

#=
Scale factor since excited slice occupies a small fraction of z FOV:
=#
# src more spins than the length of the slice
# src should I use 0.1 or 0.2 here? first was 0.2 but I changed it to 0.1 to match
zpos2_fine = range(-0.2, 0.2, 401) # z positions (cm) for 2D slice-select case
# zpos2_sim = range(-0.1, 0.1, 41) # fewer z positions (cm) for speed
zpos2_sim = range(-0.2, 0.2, 41) # fewer z positions (cm) for speed

zfov2_sim = maximum(zpos2_sim) - minimum(zpos2_sim)
zfov2_fine = maximum(zpos2_fine) - minimum(zpos2_fine)

scale2_sim = zfov2_sim / slice_width_cm / length(zpos2_sim)
scale2_fine = zfov2_fine / slice_width_cm / length(zpos2_fine)

args2_sim = (scale2_sim, rf_maker2, zpos2_sim)
args2_fine = (scale2_fine, rf_maker2, zpos2_fine)

#=
Stack real and imaginary parts:
=#
signal2_ri0(x) = real_imag(vec(signal2_c0(x)))
signal2_ri0_20(x) = real_imag(vec(signal2_c0_20(x)))
signal2_ri1(x; args=args2_sim) = real_imag(vec(signal2_c1(x, args...)));


#=
### 2D data simulations and fitting
# Simulate with noisy data with slice-profile effects,
# then fit with signal models that do/do not account for them.
=#
snr_db = 40 # set the SNR to use
yb = signal2_ri1(x; args=args2_fine) # noiseless data account for slice-profile effects
yb = reshape(yb, :, 2); yb = complex.(yb[:,1], yb[:,2]); # re-make complex!
σ = snr2sigma(snr_db, yb)
#y1 = yb + 1σ * randn(ComplexF64, size(yb));
#src @show 20*log10(norm(yb) / norm(y1 - yb))

# make a yb where have inst RF
yb_inst = signal2_ri0_20(x)
yb_inst = reshape(yb_inst, :, 2); yb_inst = complex.(yb_inst[:,1], yb_inst[:,2]); # re-make complex!
σ = snr2sigma(snr_db, yb_inst)

#=
### Slice profile effects on bSSFP
# The points show the noiseless data values that form the starting point 
# for the fit simulations.
# args:
# yb: noiseless data
# do_fine:
=#
##
function plot_bssfp_fig1(args, yb, do_fine::Bool=false;
    ylims = (0, 0.12), rtype = "sinc",
)
    α_degs = [10, 30] # flip angles
    α_rads = deg2rad.(α_degs)

    # Magnitude
    xaxis = ("Phase Cycling Factor Δθ (rad)", (-π, π), ((-1:1).*π, ["-π", "0", "π"]))
    pmism = plot( ; xaxis, ylims, widen = true, ylabel = "Signal Magnitude",
     title = "Instantaneous vs. Finite RF Two-Pool bSSFP Signal (SNR=$snr_db dB, TR=$TR_ms, TE=$TE_ms)", legend = :none, titlefontsize=20
    )
    Δϕ_plot = range(-1, 1, 401) * π # phase-cycling factors for plot
    @time tmp0 = Base.Fix{1}(_bssfp0, x).(Δϕ_plot, α_rads')
    tmp_fun(argt) = ((Δϕ, α) -> _bssfp1(x, Δϕ, α, argt...)).(Δϕ_plot, α_rads')
    @time tmp1 = tmp_fun(args); n1 = length(args[3])
    color = (1:length(α_degs))'
    if do_fine
        @time tmp2 = tmp_fun(args2_fine); 
        n2 = length(args2_fine[3]);
        labels2 = reshape(["α=$(α)°, $tRF_ms ms $rtype RF" for α in α_degs],1, :);
        plot!(Δϕ_plot, abs.(tmp2); label=labels2, color)
    end
    # labels1 = reshape(["α=$(α)°, $tRF_ms ms $rtype RF [$n1]" for α in α_degs],1, :)
    # plot!(Δϕ_plot, abs.(tmp1); label=labels1, line=:dot, color)
    labels0 = reshape(["α=$(α)°, 0 ms RF" for α in α_degs],1, :)
    plot!(Δϕ_plot, abs.(tmp0); label=labels0, line=:dash, color)
    # data_labels = reshape(["α=$(α)° noiseless data from $tRF_ms ms $rtype RF [$n2]" for α in α_degs],1, :)
    # scatter!(Δϕ_rads, abs.(reshape(yb, length(Δϕ_rads), :)); label=data_labels, color)
    
    # Phase (fairly small effect of slice-selective excitation):
    # src fix the labels on here later!!!
    pmisa = plot( ; xaxis, widen = true, ylabel = "Signal Phase (rad)",
     title = "", legend=:bottomright,legendfontsize=18
    )
    if do_fine
        plot!(Δϕ_plot, angle.(tmp2); label=labels2, color)
    end
    # plot!(Δϕ_plot, angle.(tmp1); line=:dot, label=labels1, color)
    # @show angle.(tmp0[:,1])
    # @show angle.(tmp0[:,2])
    # @show angle.(tmp0[:,3])
    # not the same this time!! should they be???
    # @assert angle.(tmp0[:,1]) ≈ angle.(tmp0[:,2]) ≈ angle.(tmp0[:,3]) # same!
    # plot!(Δϕ_plot, angle.(tmp0[:,1]); line=:dashdot, label=labels0)
    plot!(Δϕ_plot, angle.(tmp0); line=:dash, label=labels0, color)
    # plot!(Δϕ_plot, angle.(tmp0[:,1]); label="0 ms RF", line=:dashdot, color=:black)
    # scatter!(Δϕ_rads, angle.(reshape(yb, length(Δϕ_rads), :)); color, label=data_labels)

    return plot(pmism, pmisa, layout=(2,1), size=(1200, 1200),
        left_margin = 10mm, top_margin = 5mm, bottom_margin = 5mm,tickfontsize=15,guidefontsize=15,)
end
pb1 = plot_bssfp_fig1(args2_sim, yb, true)




##
##########################################################
#=
### Model fitting
Using instantaneous RF model,
despite data having slice-selection effects in the simulated data.
=#

# Function for fitting noisy signal
rand20(x::Number) = x * (1 + 0.2 * (rand() - 0.5) / 0.5) # ± 20% variability
ntry = 3


# function do_fit(signal_ri::Function, i::Int, yb, σ::Real)
#     seed!(i) # to ensure that both models fit the same data
#     y = yb + 1σ * randn(ComplexF64, size(yb)) # add noise to noiseless data yb
#     x0fun = i -> rand20.(x)
#     return fit_signal(signal_ri, x0fun, real_imag(vec(y)); ntry)
# end;
function do_fit(signal_ri::Function, i::Int, yb, σ::Real)
    seed!(i)
    y = yb + σ * randn(ComplexF64, size(yb))

    margin = 1e-8 .* max.(upper .- lower, 1.0)
    x0fun = _ -> clamp.(
        rand20.(x),
        lower .+ margin,
        upper .- margin,
    )

    return fit_signal(
        signal_ri,
        x0fun,
        real_imag(vec(y));
        ntry,
        lower,
        upper,
    ), y
end

# edited version of the above function that uses the form of
# fit_signal that takes in an array instead of a function
# it is used to initialize with the truth
function do_fit_truth(signal_ri::Function, i::Int, yb, σ::Real)
    seed!(i)
    y = yb + σ * randn(ComplexF64, size(yb))

    return fit_signal(
        signal_ri,
        x,
        real_imag(vec(y));
        ntry,
        lower,
        upper,
    ), y
end

#=
### Simulations
Because of the serious model mismatch illustrated above,
the parameter estimates are badly biased.

It seems that
the slice profile effects
(or perhaps finite RF pulse effects)
may be quite significant.
=#
#src todo: add varpro to fit_signal?

round2(x) = round(x; sigdigits=3)
mean2(x) = sum(x, dims=2)[:,1] / nrep
std2(x) = sqrt.(sum(abs2, x .- mean2(x), dims=2)[:,1] / nrep)

@show x
# error("pause here")

crb0 = sqrt.(diag(crb(signal2_ri0, x, σ))) # crb for model with inst rf
crb0_20 = sqrt.(diag(crb(signal2_ri0_20, x, σ)))
crb1_sim = sqrt.(diag(crb(signal2_ri1, x, σ))) # crb for coarse grid, finite rf
crb1_fine = sqrt.(diag(crb(x -> signal2_ri1(x; args=args2_fine), x, σ))) # crb for fine grid, finite rf
tab_crb = [:param :value :σcrb0 :σcrb1_sim :σcrb1_fine;
collect(keys(xt)) collect(xt) round2.([crb0 crb1_sim crb1_fine])
]
##
# Fit with instantaneous RF model (very fast)
# src yb is noiseless data with sp/finite rf effects, noise is added in do_fit
# src: why is this so slow right now? is not fast!! because 2 pool I think
# i need to run this!! (and edit to make it look like the one below)
# try 
#     load("/home/amurguia/Documents/mwi-data/ismrm_2027_vancouver_fits/data_sp_fit_inst_init_rand.jld")
# catch
# if !@isdefined(xr0) || false
#     nrep = 100
#     @time xr0, y_noisy0 = stack([do_fit(signal2_ri0, i, yb, σ) for i in 1:nrep])
#     mean0 = mean2(xr0)
#     std0 = std2(xr0)
#     cv0_emp = std0./mean0
#     cv0_crb = crb0./collect(xt)
#     @save "/home/amurguia/Documents/mwi-data/ismrm_2027_vancouver_fits/data_sp_fit_inst_init_rand.jld" xr0 #y_noisy0
# end;
# end
# tab2_0 = [ # estimation results table, inst RF
#  :param :value :μ0 :σ0 :σcrb0;
#  collect(keys(xt)) collect(xt) round2.([mean0 std0 crb0]);
# ]

## data with sp effects, fit with inst rf, initialize with the truth
xr0_init_truth_save_path = "/home/amurguia/Documents/mwi-data/ismrm_2027_vancouver_fits/data_sp_fit_inst_init_truth_non_zero_off_res_take2.jld"
xr0_init_truth_save_path = "/home/amurguia/Documents/mwi-data/ismrm_2027_vancouver_fits/data_sp_fit_inst_init_truth_deltaf_8Hz.jld"

nrep = 10
try
    xr0_init_truth_dict = load(xr0_init_truth_save_path)
    xr0_init_truth= xr0_init_truth_dict["xr0_init_truth"]
catch
    if !@isdefined(xr0_init_truth) || false
        @time xr0_init_truth, y_noisy0_init_truth = stack([do_fit_truth(signal2_ri0, i, yb, σ) for i in 1:nrep])
        @save xr0_init_truth_save_path xr0_init_truth y_noisy0_init_truth
    end;
end
mean0_init_truth = mean2(xr0_init_truth)
std0_init_truth = std2(xr0_init_truth)
cv0_emp_init_truth = std0_init_truth./mean0_init_truth
cv0_crb_init_truth = crb0./collect(xt) # sane as non init truth version
tab2_0_init_truth = [ # estimation results table
 :Param_Name :True_Value :Emp_Mean :Emp_SD :Emp_CV :CRB_SD :CRB_CV;
 collect(keys(xt)) collect(xt) round2.([mean0_init_truth std0_init_truth cv0_emp_init_truth crb0 cv0_crb_init_truth]);
]



## Fit with slice profile, finite RF effects (much slower, so use course z sampling)
# for 10 random starts, 3 reps this ran for about 9 hours
# if !@isdefined(xr1)
#     nrep = 3
#     xr1_list = Vector{Any}(undef, nrep)
#     y_noisy1_list = Vector{Any}(undef, nrep)
#     @time begin
#         Threads.@threads for i in 1:nrep
#             xr1_list[i], y_noisy1_list[i] = do_fit(signal2_ri1, i, yb, σ)
#             @show i
#         end
#         xr1 = stack(xr1_list)
#         y_noisy1 = stack(y_noisy1_list)
#     end
#     mean1 = mean2(xr1)
#     std1 = std2(xr1)
#     cv1_emp = std1./mean1
#     cv1_crb = crb1_sim./collect(xt)
#     @save "/home/amurguia/Documents/mwi-data/ismrm_2027_vancouver_fits/data_sp_fit_sp_init_rand.jld" xr1 y_noisy1
# end;
# tab2_1 = [ # estimation results table, slice profile/finite RF
#  :param :value :μ1 :σ1 :σcrb1;
#  collect(keys(xt)) collect(xt) round2.([mean1 std1 crb1_sim]);
# ]

## data with finite/sp effects, fit with finite/sp, initialize with the truth
xr1_init_truth_save_path = "/home/amurguia/Documents/mwi-data/ismrm_2027_vancouver_fits/data_sp_fit_sp_init_truth_fix_10reps_non_zero_off_res.jld"
xr1_init_truth_save_path = "/home/amurguia/Documents/mwi-data/ismrm_2027_vancouver_fits/data_sp_fit_sp_init_truth_fix_10reps_non_zero_off_res.jld"
nrep = 10
try
    xr1_init_truth_dict = load(xr1_init_truth_save_path)
    xr1_init_truth= xr1_init_truth_dict["xr1_init_truth"]
    y_noisy1_init_truth= xr1_init_truth_dict["y_noisy1_init_truth"]
catch
    if !@isdefined(xr1_init_truth)
        xr1_list_init_truth = Vector{Any}(undef, nrep)
        y_noisy1_list_init_truth = Vector{Any}(undef, nrep)
        @time begin
            Threads.@threads for i in 1:nrep
                @show i
                xr1_list_init_truth[i], y_noisy1_list_init_truth[i] = do_fit_truth(signal2_ri1, i, yb, σ) # diff noise instances
                @show i
            end
            xr1_init_truth = stack(xr1_list_init_truth)
            y_noisy1_init_truth = stack(y_noisy1_list_init_truth) # sane as non init truth version
        end
        @save "/home/amurguia/Documents/mwi-data/ismrm_2027_vancouver_fits/data_sp_fit_sp_init_truth_fix_10reps_non_zero_off_res.jld" xr1_init_truth y_noisy1_init_truth
    end;
end


mean1_init_truth = mean2(xr1_init_truth)
std1_init_truth = std2(xr1_init_truth)
cv1_emp_init_truth = std1_init_truth./mean1_init_truth
cv1_crb_init_truth = crb1_sim./collect(xt)
tab2_1_init_truth = [ # estimation results table, slice profile/finite RF
 :Param_Name :True_Value :Emp_Mean :Emp_SD :Emp_CV :CRB_SD :CRB_CV;
 collect(keys(xt)) collect(xt) round2.([mean1_init_truth std1_init_truth cv1_emp_init_truth crb1_sim cv1_crb_init_truth]);
]



## use data generated with inst rf (yb_inst)
# this version does not have model mismatch
# right now I'm doing it for the 3 flips and 8 PCFs, maybe add more in the future?
# change this to do parallel processing!!
ntry = 10 # 10 random starts
if !@isdefined(xr0_all_inst_rf) || false
    nrep = 25 # noise trials
    xr0_list_all_inst_rf = Vector{Any}(undef, nrep)
    y_noisy1_list_inst_rf = Vector{Any}(undef, nrep)
    @time begin
        Threads.@threads for i in 1:nrep
            xr0_list_all_inst_rf[i], y_noisy1_list_inst_rf[i] = do_fit(signal2_ri0_20, i, yb_inst, σ)
            @show i
        end
        xr0_all_inst_rf = stack(xr0_list_all_inst_rf)
        y_noisy_inst_rf = stack(y_noisy1_list_inst_rf)
    end
    mean0_inst_rf = mean2(xr0_all_inst_rf)
    std0_inst_rf = std2(xr0_all_inst_rf)
    cv0_inst_rf_emp = std0_inst_rf./mean0_inst_rf
    cv0_crb_inst_rf = crb0_20./collect(xt)
    @save "/home/amurguia/Documents/mwi-data/ismrm_2027_vancouver_fits/data_inst_fit_inst_init_rand_more_flips_more_PCFs.jld" xr0_all_inst_rf y_noisy_inst_rf
end;
tab2_0_all_inst_rf = [ # estimation results table
 :Param_Name :True_Param_Value :Empirical_Mean :Empirical_SD :Empirical_CV :CRB_SD :CRB_CV;
 collect(keys(xt)) collect(xt) round2.([mean0_inst_rf std0_inst_rf cv0_inst_rf_emp crb0_20 cv0_crb_inst_rf]);
]
##
function plot_bssfp_fit_fig2_v2()

    color = [:green :red :purple :blue :black :magenta]

    # set some initializations
    num_PCFs = length(Δϕ_rads_20)
    num_flips = length(α_rads_20)
    xaxis = ("Phase Cycling Increment Δϕ (rad)", (-π, π), ((-1:1).*π, ["-π", "0", "π"]))
    Δϕ_plot = range(-1, 1, 61) * π # fine phase-cycling factors for plot

    # raw (simulated) inst RF data
    ymat = reshape(mean2(y_noisy_inst_rf), num_PCFs, num_flips) # using noisy data
    # label_raw = reshape(map(x -> "$(x)° ideal noiseless simulated data", α_degs), 1, :)
    label_raw = reshape(map(x -> "$(x)° sim data", α_degs_20), 1, :)

    # (inst RF)
    xfit = xr0_all_inst_rf[:,1]
    signal_c0_fit_complex = signal2_c0_20(xfit) #signal2_c0(xr0_all_inst_rf)
    @show size(signal_c0_fit_complex)
    signal_c0_fit_complex_fine = Base.Fix{1}(_bssfp0, xfit).(Δϕ_plot, α_rads_20')
    
    # label_fit = reshape(map(x -> "$(x)° fitted points", α_degs_20), 1, :)
    # label_fit_fine = reshape(map(x -> "$(x)° fit", α_degs_20), 1, :)
    label_fit = reshape(
    [i == 1 ? "fitted points" : false for i in eachindex(α_degs_20)],
    1, :
)

label_fit_fine = reshape(
    [i == 1 ? "fit" : false for i in eachindex(α_degs_20)],
    1, :
)



    # magnitude (inst RF)
    pmism_inst = plot( ; xaxis, ylabel = "Signal Magnitude",widen = true,
     title = "(a) Signal Magnitude vs. Phase Cycling Factor Δθ", legend=:none)
    scatter!(Δϕ_rads_20, abs.(ymat); label=label_raw, markersize=6, color) # raw noiseless simulated data
    scatter!(Δϕ_rads_20, abs.(signal_c0_fit_complex); label=label_fit, markershape=:x,markersize=4,
     markerstrokewidth=1, color) # fitted points
    plot!(Δϕ_plot, abs.(signal_c0_fit_complex_fine); label=label_fit_fine, color) # finer fit

    # phase (inst RF)
    pmisp_inst = plot( ; xaxis, widen = true, ylabel = "Signal Phase (rad)",legend=:bottomright, title = "(b) Signal Phase vs. Phase Cycling Factor Δθ")
    scatter!(Δϕ_rads_20, angle.(ymat); label=label_raw, markersize=6, color)
    scatter!(Δϕ_rads_20, angle.(signal_c0_fit_complex); label=label_fit, markershape=:x,markersize=4, markerstrokewidth=1, color)
    plot!(Δϕ_plot, angle.(signal_c0_fit_complex_fine); label=label_fit_fine, color)

    # imaginary vs. real signal
    pmisri_inst = plot( ; xaxis, widen = true, ylabel = "Signal Phase (rad)",legend=:none, title = "(c) Imag(Signal) vs. Real(Signal)")
    scatter!(real.(ymat), imag.(ymat); label=label_raw, markersize=6, color)
    scatter!(real.(signal_c0_fit_complex), imag.(signal_c0_fit_complex); label=label_fit, markershape=:x,markersize=4, markerstrokewidth=1, color)
    plot!(real.(signal_c0_fit_complex_fine), imag.(signal_c0_fit_complex_fine); label=label_fit_fine, color)

    # ============================================================
    # Subplot 4 (v2): table1 as aligned cell annotations
    # ============================================================

    function fmt_table_cell(x; sigdigits=3)
        if x isa AbstractFloat
            if isnan(x)
                return "N/A"
            elseif isinf(x)
                return string(x)
            elseif abs(x) >= 1e4 || 0 < abs(x) < 1e-3
                return @sprintf("%.2e", x)
            else
                return string(round(x; sigdigits=sigdigits))
            end
        elseif x isa Integer
            return string(x)
        else
            return string(x)
        end
    end

    table1 = tab2_0_all_inst_rf

    headers = replace.(string.(table1[1, :]), "_" => " ")
    body = table1[2:end, :]

    nrows, ncols = size(body)

    # Table geometry
    left   = 0.02
    right  = 0.98
    top    = 0.92
    bottom = 0.08

    # Relative column widths; adjust as needed
    # col_weights = [1.45, 1.0, 1.0, 1.0, 1.0]
    # col_weights = col_weights[1:ncols]
    # col_widths = (right - left) .* col_weights ./ sum(col_weights)

    col_weights = ones(ncols)
    col_weights[1] = 1.45
    col_widths = (right - left) .* col_weights ./ sum(col_weights)

    # Column edges
    xedges = vcat(left, left .+ cumsum(col_widths))

    # Header plus body rows
    total_rows = nrows + 1
    row_height = (top - bottom) / total_rows
    yedges = top .- (0:total_rows) .* row_height

    p4 = plot(
        xlims = (0, 1),
        ylims = (0, 1),
        framestyle = :none,
        grid = false,
        legend = false,
        ticks = false,
        title = "(c) Parameter Estimates",
    )

    # Draw header and body backgrounds
    for i in 1:total_rows
        y_top = yedges[i]
        y_bottom = yedges[i + 1]

        row_color =
            i == 1      ? RGB(0.88, 0.88, 0.88) :
            iseven(i)   ? RGB(0.97, 0.97, 0.97) :
                        :white

        for j in 1:ncols
            x_left = xedges[j]
            x_right = xedges[j + 1]

            # Cell background and border
            plot!(
                p4,
                Shape(
                    [x_left, x_right, x_right, x_left],
                    [y_bottom, y_bottom, y_top, y_top]
                ),
                fillcolor = row_color,
                linecolor = :black,
                linewidth = 0.75,
                label = false
            )
        end
    end

    fontsize_header = 9
    fontsize_body = 9
    xpad = 0.008

    # Body
    for i in 1:nrows
        y = (yedges[i + 1] + yedges[i + 2]) / 2

        for j in 1:ncols
            # Horizontal center of the cell
            x = (xedges[j] + xedges[j + 1]) / 2

            cell_text = fmt_table_cell(body[i, j]; sigdigits=3)
            if j == 1
                cell_text = replace(cell_text, "_" => " ")
            end

            annotate!(
                p4,
                x,
                y,
                text(
                    cell_text,
                    fontsize_body,
                    :black,
                    :center,
                    :center
                )
            )
        end
    end

    # Headers
    for j in 1:ncols
        x = (xedges[j] + xedges[j + 1]) / 2
        y = (yedges[1] + yedges[2]) / 2

        annotate!(
            p4,
            x,
            y,
            text(
                headers[j],
                fontsize_header,
                :black,
                :center,
                :center
            )
        )
    end
    

p_combined = plot(
    pmism_inst,
    pmisp_inst,
    p4,
    layout = @layout([
        a b
        d{0.50h}
    ]),
    size = (1200, 800),
    margin = 6mm,
    bottom_margin = 8mm,
    left_margin = 8mm,
    tickfontsize = 7,
    guidefontsize = 9,
    titlefontsize = 10
)

display(p_combined)
    
end

plot_bssfp_fit_fig2_v2()


## different from above: add tables to the right (model match vs model mismatch)
# plot_bssfp_finite_rf_sp_data_fits_fig3

# helper functions

function fmt_table_cell(x; sigdigits=3)
    if x isa AbstractFloat
        if isnan(x)
            return "N/A"
        elseif isinf(x)
            return string(x)
        elseif abs(x) >= 1e4 || 0 < abs(x) < 1e-3
            return @sprintf("%.2e", x)
        else
            return string(round(x; sigdigits=sigdigits))
        end
    elseif x isa Integer
        return string(x)
    else
        return string(x)
    end
end

function table_subplot(table1; title="", fontsize_header=8, fontsize_body=8)
    headers = replace.(string.(table1[1, :]), "_" => " ")
    body = table1[2:end, :]
    nrows, ncols = size(body)

    # Table location in normalized subplot coordinates
    left   = 0.02
    right  = 0.98
    top    = 0.92
    bottom = 0.08

    # Make first column wider
    col_weights = ones(ncols)
    col_weights[1] = 1.45
    col_widths = (right - left) .* col_weights ./ sum(col_weights)
    xedges = vcat(left, left .+ cumsum(col_widths))

    total_rows = nrows + 1
    row_height = (top - bottom) / total_rows
    yedges = top .- (0:total_rows) .* row_height

    p = plot(
        xlims=(0, 1),
        ylims=(0, 1),
        framestyle=:none,
        grid=false,
        legend=false,
        ticks=false,
        title=title
    )

    # Draw cell backgrounds and borders
    for i in 1:total_rows
        ytop = yedges[i]
        ybottom = yedges[i + 1]

        row_color =
            i == 1    ? RGB(0.88, 0.88, 0.88) :
            iseven(i) ? RGB(0.97, 0.97, 0.97) :
                        RGB(1.0, 1.0, 1.0)

        for j in 1:ncols
            xleft = xedges[j]
            xright = xedges[j + 1]

            plot!(
                p,
                Shape(
                    [xleft, xright, xright, xleft],
                    [ybottom, ybottom, ytop, ytop]
                ),
                fillcolor=row_color,
                linecolor=:black,
                linewidth=0.75,
                label=false
            )
        end
    end

    # Left padding proportional to each column width
    xposition(j) =
        xedges[j] + 0.04 * (xedges[j + 1] - xedges[j])

    # Body
    for i in 1:nrows
        y = (yedges[i + 1] + yedges[i + 2]) / 2

        for j in 1:ncols
            # Horizontal center of the cell
            x = (xedges[j] + xedges[j + 1]) / 2

            cell_text = fmt_table_cell(body[i, j]; sigdigits=3)
            if j == 1
                cell_text = replace(cell_text, "_" => " ")
            end

            annotate!(
                p,
                x,
                y,
                text(
                    cell_text,
                    fontsize_body,
                    :black,
                    :center,
                    :center
                )
            )
        end
    end

    # Headers
    for j in 1:ncols
        x = (xedges[j] + xedges[j + 1]) / 2
        y = (yedges[1] + yedges[2]) / 2

        annotate!(
            p,
            x,
            y,
            text(
                headers[j],
                fontsize_header,
                :black,
                :center,
                :center
            )
        )
    end

    return p
end

function plot_bssfp_finite_rf_sp_data_fits_fig3(args::Tuple)

    # set some initializations
    num_PCFs = length(Δϕ_rads)
    num_flips = length(α_rads)
    # color = (1:length(α_degs))'
    color = [:green :red :blue]
    xaxis = ("Phase Cycling Increment Δϕ (rad)", (-π, π), ((-1:1).*π, ["-π", "0", "π"]))
    Δϕ_plot = range(-1, 1, 401) * π # fine phase-cycling factors for plot

    # raw (simulated) data
    ymat = reshape(yb, num_PCFs, num_flips)
    label_raw = reshape(map(x -> "$(x)° ideal simulated data", α_degs), 1, :)

    # fit with model mismatch (inst RF)
    signal_c0_fit_complex = signal2_c0(xr0_init_truth)
    @show size(signal_c0_fit_complex)
    signal_c0_fit_complex_fine = Base.Fix{1}(_bssfp0, xr0_init_truth).(Δϕ_plot, α_rads')
    label_fit = reshape(map(x -> "$(x)° fit points", α_degs), 1, :)
    label_fit_fine = reshape(map(x -> "$(x)° fit", α_degs), 1, :)

    # fit without model mismatch (finite RF/slice profile effects)
    signal_c1_fit_complex = signal2_c1(xr1_init_truth, args...)
    @show size(signal_c1_fit_complex)
    f = (Δϕ, α) -> _bssfp1(xr1_init_truth, Δϕ, α, args...)
    signal_c1_fit_complex_fine = f.(Δϕ_plot, α_rads')

    # magnitude (inst RF)
    pmism_inst = plot( ; xaxis, ylabel = "Signal Magnitude",widen = true,
     title = "(a) Simulated Data with SP Effects: Fit with Inst RF", legend=:none)
    scatter!(Δϕ_rads, abs.(ymat); label=label_raw, markersize=6, color) # raw noiseless simulated data
    scatter!(Δϕ_rads, abs.(signal_c0_fit_complex); label=label_fit, markershape=:x,markersize=6,
     markerstrokewidth=1, color) # fitted points
    plot!(Δϕ_plot, abs.(signal_c0_fit_complex_fine); label=label_fit_fine, color) # finer fit

    # phase (inst RF)
    pmisp_inst = plot( ; xaxis, widen = true, ylabel = "Signal Phase (rad)",legend=:none)
    scatter!(Δϕ_rads, angle.(ymat); label=label_raw, markersize=6, color)
    scatter!(Δϕ_rads, angle.(signal_c0_fit_complex); label=label_fit, markershape=:x,markersize=6, markerstrokewidth=1, color)
    plot!(Δϕ_plot, angle.(signal_c0_fit_complex_fine); label=label_fit_fine, color)

    # magnitude (finite RF/SP effects)
    pmism_sp = plot( ; xaxis, ylabel = "Signal Magnitude",widen = true,
     title = "(b) Simulated Data with SP Effects: Fit with SP Effects", legend=:none)
    scatter!(Δϕ_rads, abs.(ymat); label=label_raw, markersize=6, color) # raw simulated data
    scatter!(Δϕ_rads, abs.(signal_c1_fit_complex); label=label_fit, markershape=:x,markersize=6,
     markerstrokewidth=1, color) # fitted points
    plot!(Δϕ_plot, abs.(signal_c1_fit_complex_fine); label=label_fit_fine, color) # finer fit

    # phase (finite RF/SP effects)
    pmisp_sp = plot( ; xaxis, widen = true, ylabel = "Signal Phase (rad)",legend=:bottomright)
    scatter!(Δϕ_rads, angle.(ymat); label=label_raw, markersize=6, color)
    scatter!(Δϕ_rads, angle.(signal_c1_fit_complex); label=label_fit, markershape=:x,markersize=6, markerstrokewidth=1, color)
    plot!(Δϕ_plot, angle.(signal_c1_fit_complex_fine); label=label_fit_fine, color)

    ptable_inst = table_subplot(
    tab2_0_init_truth;
    title="(c) Parameter Estimates: Inst RF Fit",
    fontsize_header=7,
    fontsize_body=8
)

    ptable_sp = table_subplot(
    tab2_1_init_truth;
    title="(d) Parameter Estimates: SP Effects Fit",
    fontsize_header=7,
    fontsize_body=8
)

    # put all the plots on 1 grid
    p_combined = plot(
    pmism_inst,
    pmism_sp,
    pmisp_inst,
    pmisp_sp,
    ptable_inst,
    ptable_sp,
    layout=@layout([
        a b
        c d
        e{0.32h} f{0.32h}
    ]),
    size=(1400, 1200),
    margin=6mm,
    left_margin=10mm,
    bottom_margin=10mm,
    top_margin=8mm,
    tickfontsize=7,
    guidefontsize=9,
    titlefontsize=10
)

return p_combined
end

pc = plot_bssfp_finite_rf_sp_data_fits_fig3(args2_sim)

## replace table with bar graph
## Replace table with bar graph and increase all font sizes
function plot_bssfp_finite_rf_sp_data_fits_fig3_v2(args::Tuple)

    # Initializations
    num_PCFs = length(Δϕ_rads)
    num_flips = length(α_rads)

    color1 = RGB(0.000, 0.447, 0.698) # blue
    color2 = RGB(0.835, 0.369, 0.000)  # vermillion
    color3 = RGB(0.000, 0.620, 0.451) # bluish green

    color = [color1 color2 color3]

    xaxis = (
        "Phase Cycling Increment Δϕ (rad)",
        (-π, π),
        ((-1:1) .* π, ["-π", "0", "π"])
    )

    # Fine phase-cycling factors for plotting
    Δϕ_plot = range(-1, 1, 401) * π

    # Raw simulated data
    ymat = reshape(yb, num_PCFs, num_flips)

    label_raw = reshape(
        map(x -> "$(x)°", α_degs),
        1,
        :
    )

    # Fit with model mismatch: instantaneous RF
    signal_c0_fit_complex = signal2_c0(xr0_init_truth)
    @show size(signal_c0_fit_complex)

    signal_c0_fit_complex_fine =
        Base.Fix{1}(_bssfp0, xr0_init_truth).(Δϕ_plot, α_rads')

    label_fit = reshape(
    [i == 1 ? "fitted points" : "" for i in eachindex(α_degs)],
    1,
    :
)

label_fit_fine = reshape(
    [i == 1 ? "fits" : "" for i in eachindex(α_degs)],
    1,
    :
)

    # Fit without model mismatch: finite RF/slice-profile effects
    signal_c1_fit_complex =
        signal2_c1(xr1_init_truth, args...)

    @show size(signal_c1_fit_complex)

    f = (Δϕ, α) ->
        _bssfp1(xr1_init_truth, Δϕ, α, args...)

    signal_c1_fit_complex_fine =
        f.(Δϕ_plot, α_rads')

    # Magnitude: instantaneous RF
    pmism_inst = plot(
        ;
        xaxis,
        ylabel="Signal Magnitude",
        widen=true,
        title="(a) Simulated Data with SP Effects: Fit with Inst RF",
        legend=:none
    )

    scatter!(
        pmism_inst,
        Δϕ_rads,
        abs.(ymat);
        label=label_raw,
        markersize=6,
        color
    )

    scatter!(
        pmism_inst,
        Δϕ_rads,
        abs.(signal_c0_fit_complex);
        label=label_fit,
        markershape=:x,
        markersize=6,
        markerstrokewidth=1,
        color
    )

    plot!(
        pmism_inst,
        Δϕ_plot,
        abs.(signal_c0_fit_complex_fine);
        label=label_fit_fine,
        color
    )

    # Phase: instantaneous RF
    pmisp_inst = plot(
        ;
        xaxis,
        widen=true,
        ylabel="Signal Phase (rad)",
        legend=:none
    )

    scatter!(
        pmisp_inst,
        Δϕ_rads,
        angle.(ymat);
        label=label_raw,
        markersize=6,
        color
    )

    scatter!(
        pmisp_inst,
        Δϕ_rads,
        angle.(signal_c0_fit_complex);
        label=label_fit,
        markershape=:x,
        markersize=6,
        markerstrokewidth=1,
        color
    )

    plot!(
        pmisp_inst,
        Δϕ_plot,
        angle.(signal_c0_fit_complex_fine);
        label=label_fit_fine,
        color
    )

    # Magnitude: finite RF/slice-profile effects
    pmism_sp = plot(
        ;
        xaxis,
        ylabel="Signal Magnitude",
        widen=true,
        title="(b) Simulated Data with SP Effects: Fit with SP Effects",
        legend=:none
    )

    scatter!(
        pmism_sp,
        Δϕ_rads,
        abs.(ymat);
        label=label_raw,
        markersize=6,
        color
    )

    scatter!(
        pmism_sp,
        Δϕ_rads,
        abs.(signal_c1_fit_complex);
        label=label_fit,
        markershape=:x,
        markersize=6,
        markerstrokewidth=1,
        color
    )

    plot!(
        pmism_sp,
        Δϕ_plot,
        abs.(signal_c1_fit_complex_fine);
        label=label_fit_fine,
        color
    )

    # Phase: finite RF/slice-profile effects
    pmisp_sp = plot(
        ;
        xaxis,
        widen=true,
        ylabel="Signal Phase (rad)",
        legend=:bottomright
    )

    scatter!(
        pmisp_sp,
        Δϕ_rads,
        angle.(ymat);
        label=label_raw,
        markersize=6,
        color
    )

    scatter!(
        pmisp_sp,
        Δϕ_rads,
        angle.(signal_c1_fit_complex);
        label=label_fit,
        markershape=:x,
        markersize=6,
        markerstrokewidth=1,
        color
    )

    plot!(
        pmisp_sp,
        Δϕ_plot,
        angle.(signal_c1_fit_complex_fine);
        label=label_fit_fine,
        color
    )

    # Grouped bar graph
    function parameter_error_barplot(tab_inst, tab_finite)

        # Columns:
        #   1: parameter
        #   2: true value
        #   3: estimated mean
        parameters =
            replace.(string.(tab_inst[2:end, 1]), "_" => " ")

        finite_parameters =
            replace.(string.(tab_finite[2:end, 1]), "_" => " ")

        parameters == finite_parameters ||
            error(
                "Tables must contain the same parameters " *
                "in the same order."
            )

        percent_error(tab) = [
            iszero(t) ? NaN : 100 * abs(m - t) / abs(t)
            for (t, m) in zip(
                Float64.(tab[2:end, 2]),
                Float64.(tab[2:end, 3])
            )
        ]

        inst_error = percent_error(tab_inst)
        finite_error = percent_error(tab_finite)

        x = collect(eachindex(parameters))
        offset = 0.13
        bar_width = 0.22

        finite_rf_color = RGB(0.35, 0.35, 0.35)     # dark gray
        inst_rf_color   = RGB(0.800, 0.475, 0.655)  # reddish purple

        p = bar(
    x .- offset,
    finite_error;
    bar_width=bar_width,
    color=finite_rf_color,
    linecolor=:black,
    linewidth=2,
    label="Finite RF/SP fit"
)

bar!(
    p,
    x .+ offset,
    inst_error;
    bar_width=bar_width,
    color=inst_rf_color,
    linecolor=:black,
    linewidth=2,
    label="Inst RF fit"
)

        plot!(
            p;
            xticks=(x, parameters),
            xlabel="Parameter",
            ylabel="Absolute Percent Error (%)",
            title="(c) Mean Estimate Error Relative to True Value",
            legend=:topright,
            xrotation=45,
            xlims=(0.4, length(parameters) + 0.6),
            widen=false,

            # Larger bar-plot fonts
            tickfontsize=11,
            guidefontsize=13,
            titlefontsize=14,
            legendfontsize=9
        )

        return p
    end

    pmean_abs_diff = parameter_error_barplot(
        tab2_0_init_truth,
        tab2_1_init_truth
    )

    # Combined figure
    p_combined = plot(
        pmism_inst,
        pmism_sp,
        pmisp_inst,
        pmisp_sp,
        pmean_abs_diff;
        layout=@layout([
            a b
            c d
            e{0.32h}
        ]),
        size=(1600, 1400),
        margin=8mm,
        left_margin=14mm,
        bottom_margin=14mm,
        top_margin=10mm,

        # Larger fonts throughout
        tickfontsize=13,
        guidefontsize=15,
        titlefontsize=16,
        legendfontsize=13
    )

    return p_combined
end

pc_v2 = plot_bssfp_finite_rf_sp_data_fits_fig3_v2(args2_sim)


