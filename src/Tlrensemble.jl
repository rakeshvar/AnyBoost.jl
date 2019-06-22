export LREnsemble
export trainerr, trainerrseq, testerr, testerrseq
export normcardseq

mutable struct LREnsemble{F, B<:AbBasis{F}}
    prob::Problem
    ωs::Vector{B}           # Predictor Parameters
    βs::Matrix{F}
    zs::Vector{Vector{F}}   # Predictors
    fit::Vector{F}
    λs::Vector{F}           # Covariances
    δβ::Vector{F}           # Change of dir of β
    η::Vector{F}            # Equiangular direction
    As::Vector{F}           # Equiangles
    γs::Vector{F}           # Arclengths
    Gram::Matrix{F}         # Gram Matrix
    active::BitVector
    γzerohit::Vector{F}
end

function LREnsemble(prob::Problem{MSELoss, B},
                    ω::B,
                    z::Vector{F}=calc(ω, prob.X),
                    C::F=dot(prob.y, z)) where {F, B<:AbBasis{F}}
    normz = norm(z)
    LREnsemble{F, B}(
        prob,
        [ω],
        hcat(0.),
        [z],
        zeros(prob.y),
        [C],
        [1./normz],
        z/normz,
        [normz],
        [0.],
        hcat(normz^2),
        trues(1),
        [Inf])
end

clone(e::LREnsemble) =
    LREnsemble(
        e.prob,
        deepcopy(e.ωs),
        copy(e.βs),
        deepcopy(e.zs),
        copy(e.fit),
        copy(e.λs),
        copy(e.δβ),
        copy(e.η),
        copy(e.As),
        copy(e.γs),
        copy(e.Gram),
        copy(e.active),
        copy(e.γzerohit))

################################################################################
#
################################################################################

function mindist(ωs::Vector{B}, u::Vector{F}, s::F, active::BitVector=trues(length(ωs))) where {F, B<:AbBasis{F}}
    d = Inf
    for (ω, actv) in zip(ωs, active)
        if ω.sgn==s && actv
            d = min(d, norm(u-ω.u))
        end
    end
    d
end
mindist(ωs::Vector{B}, ω::B) where {F, B<:AbBasis{F}} = mindist(ωs, ω.u, ω.sgn)
mindist(ens::LREnsemble{F, B}, u::Vector{F}, s::F) where {F, B<:AbBasis{F}} = mindist(ens.ωs, u, s)#, ens.active)
mindist(ens::LREnsemble{F, B}, ω::B) where {F, B<:AbBasis{F}} = mindist(ens.ωs, ω.u, ω.sgn)#, ens.active)
mindist(us::Vector{Vector{F}}, u::Vector{F}) where F = (length(us)==0)?Inf:minimum(norm(u-ui) for ui in us)
dists(ens::LREnsemble, na=-1.) = [ω1.sgn==ω2.sgn?norm(ω1.u-ω2.u):na for ω1 in ens.ωs, ω2 in ens.ωs]

################################################################################
#
################################################################################

function normcardseq(ens::LREnsemble)
    mapslices(β->norm(β, 1), ens.βs, 1)[:], mapslices(countnz, ens.βs, 1)[:]
end

trainerr(ens::LREnsemble, los=ens.prob.loss) = los(ens.prob.y, ens.fit)
testerr(ens::LREnsemble, X::Matrix, y::Vector, los=ens.prob.loss) = los(y, sum(ens.βs[:, end] .* calc.(ens.ωs, (X,))))

function trainerrseq(ens::LREnsemble{F}, los=ens.prob.loss) where F
    fits = hcat(ens.zs...)*ens.βs
    [los(ens.prob.y, fits[:, i]) for i in 1:size(fits)[2]]
end

function testerrseq(ens::LREnsemble{F}, X::Matrix{F}, y::Vector{F}, los=ens.prob.loss) where F
    zs = hcat(calc.(ens.ωs, (X,))...)
    fits = zs*ens.βs
    [los(y, fits[:, i]) for i in 1:size(fits)[2]]
end

################################################################################
#
################################################################################

function comparecatchup(ens::LREnsemble{F}, zold::Vector{F}, znews::Vector{Vector{F}}) where F<:AbstractFloat
    C, A = ens.λs[end], ens.As[end]
    r = ens.prob.y-ens.fit
    Cold = r'*zold
    Cnews = [r'*znew for znew in znews]
    Aold = ens.η'*zold
    Anews = [ens.η'*znew for znew in znews]
    γold = (C-Cold)/(A-Aold)
    γnews = [(C-Cnews[i])/(A-Anews[i]) for i in 1:length(znews)]
    printfmtln("\t𝒜  :C={:6.4f} A={:6.4f} ", C, A)
    printfmtln("\tOLD:C={:6.4f} A={:6.4f} γ={:6.4f} C(γ)={:6.4f}", Cold, Aold, γold, C-γold*A)
    for i in 1:length(znews)
    printfmtln("\tNEW:C={:6.4f} A={:6.4f} γ={:6.4f} C(γ)={:6.4f}", Cnews[i], Anews[i], γnews[i], C-γnews[i]*A)
    end
end

comparecatchup(ens::LREnsemble{F, B}, zold::Vector{F}, ωnews::Vector{B}) where {F, B<:AbBasis{F}} =
    comparecatchup(ens, zold, [calc(ω, ens.prob.X) for ω in ωnews])

function printens(ens::LREnsemble)
    println("LARS Ensemble")
    println("\tLength ", length(ens.ωs))
    println("\tLoss ", ens.prob.loss(ens.prob.y, ens.fit))
    println("\tλs\t", ens.λs)
    println("\tC(u)\t", [(ens.prob.y-ens.fit)'*z for z in ens.zs])
    println("\tAs\t", ens.As)
    println("\tA(u)\t", [ens.η'*z for z in ens.zs])
    println("\tγs\t", ens.γs)
    pretty("\tβs ", ens.βs)
    println("\tδβ\t", ens.δβ)
    println("\tγ0\t", ens.γzerohit)
    println("\t𝒜\t", ens.active*1)
    println("\tSigns\t", [ω.sgn>0?"+":"-" for ω in ens.ωs])
    pretty("\tGram ", ens.Gram)
    pretty("\tDists ", dists(ens, 1e-3), 3)
    printfmtln("\tClosest\t{:7.5f}", minimum([((ω1.sgn==ω2.sgn)&&(ω1.u!=ω2.u))?norm(ω1.u-ω2.u):Inf for ω1 in ens.ωs, ω2 in ens.ωs]))
end

function pretty(name, M::Matrix{F}, dig=4) where F<:AbstractFloat
    s = reprmime("text/plain", round.(Int, M*10.0^dig))
    s = s[findfirst("\n", s)[1]:end]
    s = replace(s, "\n"=>"\n\t")
    println(name*s)
end
