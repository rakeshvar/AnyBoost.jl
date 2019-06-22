using AnyBoost
using Plots
using DataFrames
include("xgb.jl")

function check(args, N::Int, types...)
    @assert length(args) == N
    @assert length(types) == N
    for i in 1:N
        @assert isa(args[i], types[i])
    end
end

function onerun(trdata, tedata, l, B, O, algo, args, kwargs)
    if algo === :xg
        check(args, 0)
        return xgb(trdata, tedata, l, args...; kwargs...)
    end

    prob = Problem(trdata.X, trdata.y, l, B, O)

    time = @elapsed if algo === :fw
        check(args, 1, Vector)
        ens = ADFWPath(prob, args...; kwargs...)
    elseif algo === :gb
        check(args, 1, Real)
        ens = GradientBoost(prob, args...; kwargs...)
    elseif algo === :sw
        check(args, 1, Real)
        ens = Stagewise(prob, args...; kwargs...)
    elseif algo === :nn
        check(args, 2, Vector, Int)
        ens = NNPath(prob, args...; kwargs...)
    else
        error("Unknown Algorithm: ", algo)
    end

    norms, cards = normcardseq(ens)

    trerrs = trainerrseq(ens)/trdata.n
    teerrs = testerrseq(ens, tedata.X, tedata.y)/tedata.n
    bayes = l(tedata.y, tedata.signal)/tedata.n

    al = altloss(l)
    altrerrs = trainerrseq(ens, al)/trdata.n
    alteerrs = testerrseq(ens, tedata.X, tedata.y, al)/tedata.n
    albayes = al(tedata.y, tedata.signal)/tedata.n

    cards, norms, trerrs, teerrs, bayes, altrerrs, alteerrs, albayes, time
end

function multirun(datagetterfn,
        algo::Symbol, nruns::Int, l::AbLoss,
        B::Type{<:AbBasis},
        O::Type{<:AbConstraintSet},
        args::Tuple, kw::Dict)
    F = Float64
    df = DataFrame(Run=Int[], Iteration=Int[],
                   Regularization=F[],
                   Cardinality=Int[], Norm=F[],
                   Trainerr=F[], Testerr=F[], Bayes=F[],
                   Altrainerr=F[], Altesterr=F[], AltBayes=F[],
                   Time=F[])

    for i in 1:nruns
        println("######################## Running Iteration $i ########################")
        trdata, tedata = datagetterfn(i)
        try
            cards, norms, trerrs, teerrs, bayes, altrerrs, alteerrs, albayes, time = onerun(trdata, tedata, l, B, O, algo, args, kw)
            regularization = (algo in (:fw, :nn)) ? args[1] : 1:length(trerrs)
            for j in 1:length(regularization)
                dfentry = [i, j, regularization[j], cards[j], norms[j],
                            trerrs[j], teerrs[j], bayes,
                            altrerrs[j], alteerrs[j], albayes, time]
                push!(df, dfentry)
                println(dfentry)
                flush(STDOUT)
            end
        catch e
            println(sprint(showerror, e, catch_backtrace()))
        end
    end

    return df
end

function multirunsimulated(;
    algo::Symbol=:fw,
    nruns::Int=25,
    nfuncs::Int=20, pgood::F=.5,
    p::Int=31, snr::F=1., ρ::F=.3,
    ntr::Int=5000, nte::Int=5000,
    l::AbLoss=MSE,
    B=Relu{F},
    O=L2Constraint(),
    args::Tuple=(),
    kw::Dict{Symbol, Anny}=Dict{Symbol, Any}()) where {F<:AbstractFloat, Anny}

    function getdata(i)
        srand(1000_000+1000i)
        gendata(argaussianDM(p, FunctionMechanism(nfuncs, pgood), snr, ρ), ntr, nte, binary=l==LOGIT)
    end
    multirun(getdata, algo, nruns, l, B, O, args, kw)
end

function multirunreal(datax, datay;
    algo::Symbol=:fw,
    ncvs::Int=10,
    l::AbLoss=MSE,
    B::Type{<:AbBasis}=Relu{Float64},
    O::Type{<:AbConstraintSet}=L2Constraint,
    args::Tuple=(),
    kw::Dict=Dict{Symbol, Any}())

    n, p = size(datax)
    srand(02052019)
    testfolds = collect(Iterators.partition(randperm(n), Int(n/ncvs)))
    trainfolds = [setdiff(1:n, testfold) for testfold in testfolds]
    function getdata(icv)
        (RealData(datax[trainfolds[icv], :], datay[trainfolds[icv]]),
         RealData(datax[testfolds[icv], :], datay[testfolds[icv]]))
    end
    multirun(getdata, algo, ncvs, l, B, O, args, kw)
end

function singlerunreal(datax, datay, dataoutx, dataouty;
    algo::Symbol=:fw,
    ncvs::Int=0,
    l::AbLoss=MSE,
    B::Type{<:AbBasis}=Relu{F},
    O::Type{<:AbConstraintSet}=L2Constraint,
    args::Tuple=(),
    kw::Dict=Dict{Symbol, Any}())

    function getdata(icv)
        @assert icv===1
        (RealData(datax, datay), RealData(dataoutx, dataouty))
    end
    multirun(getdata, algo, 1, l, B, O, args, kw)
end
