export AbEnsemble, BBEnsemble, GBEnsemble, FWEnsemble, SWEnsemble
export trainerr, trainerrseq, testerr, testerrseq, prediction
export normωs, normcardseq, getβseq

abstract type AbEnsemble{F, B} end

#################
# BareBones
#################
struct BBEnsemble{F, B<:AbBasis{F}} <:  AbEnsemble{F, B}
    prob::Problem
    ωs::Vector{B}
    β::Vector{F}
end

BBEnsemble(ens::AbEnsemble{F,B}) where {F, B<:AbBasis{F}} =
        BBEnsemble{F, B}(ens.prob, deepcopy(ens.ωs), copy(ens.β))

#################
# Gradient Boost
#################
mutable struct GBEnsemble{F, B<:AbBasis{F}} <:  AbEnsemble{F, B}
    prob::Problem
    ωs::Vector{B}
    β::Vector{F}
    zs::Vector{Vector{F}}
    fit::Vector{F}
end

GBEnsemble(prob::Problem{L,B}) where {L, F, B<:AbBasis{F}} =
        GBEnsemble{F, B}(prob, B[], F[], Vector{F}[], zeros(F, sizen(prob)))

#################
# Frank Wolfe
#################
mutable struct FWEnsemble{F, B<:AbBasis{F}} <:  AbEnsemble{F, B}
    prob::Problem
    ωs::Vector{B}
    β::Vector{F}
    zs::Vector{Vector{F}}
    fit::Vector{F}
end

FWEnsemble(prob::Problem{L,B}) where {L, F, B<:AbBasis{F}} =
        FWEnsemble{F, B}(prob, B[], F[], Vector{F}[], zeros(F, sizen(prob)))

#################
# Basics
#################
import Base.length
length(ens::AbEnsemble) = length(ens.ωs)

import Base.push!
function push!(ens::AbEnsemble{F, B}, ω::B, βi::F, z::Vector{F}) where {B, F}
    push!(ens.ωs, ω)
    push!(ens.β, βi)
    push!(ens.zs, z)
    ens.fit += βi * z
    ens
end

push!(ens::AbEnsemble{F, B}, ω::B, βi::F) where {B, F} = push!(ens, ω, βi, calc(ω, ens.prob.X))
push!(ens::AbEnsemble{F, B}, ω::B) where {B, F} = push!(ens, ω, zero(F))

getZ(ens::AbEnsemble) = hcat(ens.zs...)

function _correctfit!(ens::AbEnsemble)
    ens.fit = sum(ens.β .* ens.zs)
end

function _correctzs!(ens::AbEnsemble)
    for j in 1:length(ens)
        ens.zs[j] = calc(ens.ωs[j], ens.prob.X)
    end
    _correctfit!(ens)
end

#################
# {ω} updates for Backprop
#################
function ssedir(ens1::AbEnsemble{F, B}, ens2::AbEnsemble{F, B}) where {F, B<:AbUniDirBasis{F}}
    @assert length(ens1) ==  length(ens2)
    sum(sum(abs2, ens1.ωs[j].u - ens2.ωs[j].u) for j in 1:length(ens1))
end

function stepalong!(ens::AbEnsemble{F, B}, ∇L_u, t) where {F, B<:AbUniDirBasis{F}}
    @assert length(ens) == size(∇L_u)[2]
    for j in 1:length(ens)
        ens.ωs[j].u -= t*∇L_u[:, j]                 # Negative gradient
        ens.ωs[j].u = project(ens.prob.Ω, ens.ωs[j].u)
    end
    _correctzs!(ens)
end

#################
# FW β updates
#################
function updateβ!(ens::FWEnsemble{F}, β::Vector{F}, flush=true) where F #lassoonA
    @assert length(β)==length(ens.β)
    keep = flush ? (β.!= 0.0) : (1:length(β))
    ens.β = β[keep]
    if flush
        ens.ωs = ens.ωs[keep]
        ens.zs = ens.zs[keep]
    end
    _correctfit!(ens)
    ens
end

function fwvanillaupdateβ!(ens::FWEnsemble, iter::Int, C::AbstractFloat)
    @assert ens.β[end] == 0.
    oldfactor = iter/(iter+2.)

    ens.β *= oldfactor
    ens.β[end] = C * 2./(iter+2.)

    ens.fit *= oldfactor
    ens.fit += ens.β[end] * ens.zs[end]
end

################
# Stagewise
################
mutable struct SWEnsemble{F, B<:AbBasis{F}} <:  AbEnsemble{F, B}
    prob::Problem
    ωs::Vector{B}
    β::Vector{F}
    zs::Vector{Vector{F}}
    fit::Vector{F}
    ϵ::F
    δ::F
    events::Vector{Int}
    _βs::Matrix{F}
end

SWEnsemble(prob::Problem{L,B}, ϵ::F, δ::F) where {L, F, B<:AbBasis{F}} =
        SWEnsemble{F, B}(prob, B[], F[], Vector{F}[], zeros(F, sizen(prob)), ϵ, δ, Int[], Matrix{F}(0, 0))

function pushnew!(ens::SWEnsemble{F, B}, ω::B, z::Vector{F}=calc(ω, ens.prob.X)) where {B, F}
    push!(ens.ωs, ω)
    push!(ens.β, ens.ϵ)
    push!(ens.zs, z)
    ens.fit += ens.ϵ * z
    push!(ens.events, length(ens.β))
    ens
end

function decrementβi!(ens::SWEnsemble, i::Int)
    tol = 1e-9
    @assert ens.β[i] > tol
    if ens.β[i] < ens.δ + tol       # Clamp down to zero
        ens.fit -= ens.β[i] * ens.zs[i]
        ens.β[i] = 0.
    else
        ens.fit -= ens.δ * ens.zs[i]
        ens.β[i] -= ens.δ
    end
    push!(ens.events, -i)
end

function augmentβi!(ens::SWEnsemble, i::Int)
    ens.fit += ens.ϵ * ens.zs[i]
    ens.β[i] += ens.ϵ
    push!(ens.events, i)
end

function _fillβs(ens::SWEnsemble{F}) where F
    # Each Column is a variable, time progresses along a column length
    βs = zeros(length(ens.events)+1, maximum(ens.events))
    k = 1
    for ivar in ens.events
        k += 1
        if ivar > 0
            βs[k:end, ivar] += ens.ϵ
        else
            ivar = abs(ivar)
            if βs[k, ivar] > ens.δ + 1e-9
                βs[k:end, ivar] -= ens.δ
            else
                βs[k:end, ivar] = zero(F)
            end
        end
    end
    @assert βs[end, :] ≈ ens.β
    ens._βs = βs
end

function ensureβfilled(ens::SWEnsemble{F}) where F
    ((length(ens._βs) > 0) && (ens._βs[end, :] ≈ ens.β)) || _fillβs(ens)
end

function getβseq(ens::SWEnsemble{F}, fillval::F=NaN) where F
    ensureβfilled(ens)
    (fillval == 0.) && return ens._βs
    @assert fillval===NaN
    βs = fill(NaN, size(ens._βs))
    for ivar in 1:size(βs)[2]
        a = max(1, findfirst(ens._βs[:, ivar])-1)
        b = min(findlast(ens._βs[:, ivar])+1, size(βs)[1])
        βs[a:b, ivar] = ens._βs[a:b, ivar]
    end
    βs
end

function getβseq(enses::Vector{E}, fillval::F=NaN) where {F, E<:BBEnsemble{F}}
    βs = fill(fillval, (length(enses), length(enses[end])))
    for (itime, ens) in enumerate(enses)
        βs[itime, 1:length(ens)] = ens.β
    end
    βs
end

################################################################################
# Exports
################################################################################
normωs(ens::AbEnsemble{F, B}) where {F, B<:AbUniDirBasis{F}} = [normcs(ens.prob, ω.u) for ω in ens.ωs]
trainerr(ens::AbEnsemble, los=ens.prob.loss) = los(ens.prob.y, ens.fit)
testerr(ens::AbEnsemble, X::Matrix, y::Vector, los=ens.prob.loss) = los(y, sum(ens.β .* calc.(ens.ωs, (X,))))
trainerr(ens::BBEnsemble, los=ens.prob.loss) = testerr(ens, ens.prob.X, ens.prob.y, los)

# Vector of Ensembles
#################
trainerrseq(enses::Vector{E}, los=enses[1].prob.loss) where E<:AbEnsemble = trainerr.(enses, los)
testerrseq(enses::Vector{E}, X::Matrix, y::Vector, los=enses[1].prob.loss) where E<:AbEnsemble = testerr.(enses, (X,), (y,), los)
normcardseq(enses::Vector{E}) where E<:AbEnsemble = [norm(ens.β, 1) for ens in enses], [countnz(ens.β) for ens in enses]

# GBEnsembles
#################
trainerrseq(ens::GBEnsemble, los=ens.prob.loss) = los.((ens.prob.y,),  cumsum(ens.β .* ens.zs))
testerrseq(ens::GBEnsemble, X::Matrix, y::Vector, los=ens.prob.loss) = los.((y,),  cumsum(ens.β .* calc.(ens.ωs, (X,))))
normcardseq(ens::GBEnsemble) = cumsum(abs.(ens.β)), 1:length(ens)

# SWEnsembles
#################
function normcardseq(ens::SWEnsemble{F}) where F
    ensureβfilled(ens)
    mapslices(β->norm(β, 1), ens._βs, 2)[:], mapslices(countnz, ens._βs, 2)[:]
end

function _swerrseq(ens::SWEnsemble{F},
                     y::Vector{F},
                     zs::Vector{Vector{F}},
                     los
                     ) where F
    losses = zeros(length(ens.events)+1)
    z = zeros(zs[1])
    losses[1] = los(y, z)

    ϵzs = ens.ϵ * zs
    for (k, i) in enumerate(ens.events)
        if i > 0
            z += ϵzs[i]
        else
            z -= ens.δ * zs[abs(i)]       # Not checking for Negative β
        end
        losses[k+1] = los(y, z)
    end
    losses
end

trainerrseq(ens::SWEnsemble{F}, los=ens.prob.loss) where F = _swerrseq(ens, ens.prob.y, ens.zs, los)
testerrseq(ens::SWEnsemble{F}, X::Matrix{F}, y::Vector{F}, los=ens.prob.loss) where F = _swerrseq(ens, y, calc.(ens.ωs, (X,)), los)

#################
# Checks
#################
function sanitycheck(ens::AbEnsemble)
    @assert length(ens.ωs) == length(ens.β) == length(ens.zs)
    @assert length(ens.fit) == sizen(ens.prob)
    @assert all(length.(ens.zs) .== length(ens.fit))
    @assert (length(ens) == 0 && sum(abs, ens.fit) == 0) || (ens.fit ≈ sum(ens.β .* ens.zs)) "$(length(ens)) $(sum(abs, ens.fit)) $(ens.fit ≈ sum(ens.β .* ens.zs))"
    @assert all(z ≈ calc(ω, ens.prob.X) for (z, ω) in zip(ens.zs, ens.ωs))
    print("✓")
end
