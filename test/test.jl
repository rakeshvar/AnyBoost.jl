include("loadrealdata.jl")
include("run.jl")
using DataFrames

F = typeof(1.0)

activations = [
    Relu{F},
    Requ{F}
]

constraintsets = [
    L2Constraint,
    BasisL2Constraint
]

Cs = [1e-3, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 100]

algconf(algo::Symbol, args::Tuple, kw::Dict) = Dict(:algo=>algo, :args=>args, :kwargs=>kw)
algconfigs = [algconf(a...) for a in (
    (:fw, (Cs,), Dict(:ad=>false)),
    (:fw, (Cs,), Dict(:ad=>true)),

    (:gb, (1.,), Dict(:ad=>false, :maxiter=>800)),
    (:gb, (1.,), Dict(:ad=>true, :maxiter=>800)),

    (:sw, (1/16.,), Dict(:ad=>false, :maxiter=>1000)),
    (:sw, (1/16.,), Dict(:ad=>true, :maxiter=>1000)),

    (:sw, (1/16.,), Dict(:lasso=>true, :maxiter=>1500)),
    (:sw, (1/16.,), Dict(:lasso=>true, :ad=>true, :maxiter=>1500)),

    (:nn, (Cs, 300), Dict(:stepsize=>3e-3, :maxiter=>5000)),
    (:xg, (), Dict(:numround=>500)),
)]

function getarg(idx, typ::Type{T}=Int, default::T=1) where T
    if length(ARGS) ≥ idx
        return parse(ARGS[idx])::T
    else
        return default
    end
end

dataidx = getarg(1, Symbol, :toy1)
algidx = getarg(2)
actidx  = getarg(4)
considx = getarg(3)
if length(ARGS) == 0
    println("Usage:\n\ttest.jl [dataset_idx=:toy1] [algorithm_idx=1] [activation_idx=1] [constraint_set_idx=1]")
    println("Using defaults...")
end

trdata, tedata = loadrealdata(dataidx)
l = all(tedata.y.^2 .== tedata.y) ? LOGIT : MSE
B = activations[actidx]
O = constraintsets[considx]
algconfig = algconfigs[algidx]

##
println("Loss: ", l)
println("Activation: ", B)
println("Constraint Set: ", O)
println("Algo: ", algconfig)
println("Data: ", dataidx)
##

r = onerun(trdata, tedata, l, B, O; algconfig...)
@show DataFrame(Iteration=1:length(r[1]),
               Cardinality=r[1], Norm=r[2],
               Trainerr=r[3], Testerr=r[4], Bayes=r[5],
               Altrainerr=r[6], Altesterr=r[7], AltBayes=r[8],
               Time=r[9])
