
#############################
# Wrappers for easy calling
#############################
linesearch(prob::Problem{L}, z::Vector{T}, v::Vector{T}) where {T<:Real, L} =
    linesearch(L, prob.y, z, v)

linesearch(prob::Problem{L, B, CS}, z::Vector{T}, ω::B) where {T<:Real, L, B, CS} =
    linesearch(L, prob.y, z, predictor(prob, ω))

linesearch(prob::Problem{L, B}, ωs::Vector{B}, ω::B) where {L, B} =
    linesearch(L, prob.y, prediction(prob, ωs), ω)

#############################
# For various Losses
#############################
linesearch(::Type{L}, args...) where {L<:AbLoss} = error("Not Implemented.")

linesearch(::Type{MSELoss}, y, z, v) = ((y-z)'*v)/(v'*v)

function linesearch(::Type{LogitLoss},
                    y::Vector{T},
                    z::Vector{T},
                    v::Vector{T},
                    ) where T<:Real
    ρ = zero(T)
    for i in 1:10
        s = sigmoid.(z+ρ*v)
        grad = v'*(s-y)
        hess = v'*(v.*s.*(1-s))
        dec = grad/hess
        ρ -= dec
        @debug "$i) $dec $ρ"
        grad*dec < 1e-6 && break         # Newton Decrement
    end
    ρ
end

function weighted_median(data::Vector{T1}, weights::Vector{T})::T1 where {T1,T<:Real}
    ordered = sortperm(data)
    cumordwts = cumsum(weights[ordered])
    data[ordered][findfirst(cumordwts .≥ cumordwts[end]/2)]
end

linesearch(::Type{MADLoss}, y, z, v) = weighted_median((y-z)./v, abs.(v))
