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

function onerun(trdata, tedata, l, B, O; algo=:fw, args=([1., 2.]), kwargs=Dict())
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
