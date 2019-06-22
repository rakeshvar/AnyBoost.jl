export AbBasis, LinearUnit, Relu, Requ

abstract type AbBasis{F<:AbstractFloat} end
abstract type AbUniDirBasis{F}<:AbBasis{F} end
calc{F<:AbUniDirBasis}(::Type{F}, X, u) = calc(F, X*u)

################################################################# Linear
struct LinearUnit{F} <: AbUniDirBasis{F}
  u::Vector{F}
end
calc{F}(::Type{LinearUnit{F}}, x) = x
derv{F}(::Type{LinearUnit{F}}, x) = ones(x)
calc{F}(b::LinearUnit{F}, X) = X*b.u

################################################################# Signed
abstract type AbSignedUniDirBasis{F}<:AbUniDirBasis{F} end
calc(b::B, X) where B<:AbSignedUniDirBasis = b.sgn * calc(B, X*b.u)
derv(b::B, X) where B<:AbSignedUniDirBasis = b.sgn * derv.(B, X*b.u)

################################################################# Relu
mutable struct Relu{F} <: AbSignedUniDirBasis{F}
  u::Vector{F}
  sgn::F
end
Relu{F}(u::Vector{F}) = Relu(u, one(F))
calc{F}(::Type{Relu{F}}, x) = max.(x, 0.)
derv{F}(::Type{Relu{F}}, x) = (x .> 0.) * 1.

################################################################# ReQU
struct Requ{F} <: AbSignedUniDirBasis{F}
  u::Vector{F}
  sgn::F
end
Requ{F}(u::Vector{F}) = Requ(u, one(F))
calc{F}(::Type{Requ{F}}, x) = max.(x, 0) .^ 2
derv{F}(::Type{Requ{F}}, x) = 2. * (x.>0) .* x

################################################################# Huber
struct Huber{t, F} <: AbSignedUniDirBasis{F}
    u::Vector{F}
    sgn::F
end
Huber{t}(u::Vector{F}, s::F) where {t, F} = Huber{t, F}(u, s)
Huber{t}(u::Vector{F}) where {t, F} = Huber{t, F}(u, one(F))
Huber(u::Vector{F}, s::F) where {F} = Huber{1., F}(u, s)
Huber(u::Vector{F}) where {F} = Huber{1., F}(u, one(F))
calc(::Type{Huber{t, F}}, x) where {F, t} = (x.>0) .* ((x*t .< 1) .* (t*x.*x/2) + (x*t .≥ 1) .* (x-.5/t))
derv(::Type{Huber{t, F}}, x) where {F, t} = (x.>0) .* ((x*t .< 1) .* (t*x)      + (x*t .≥ 1)            )

################################################################# Softplus
struct Softplus{t, F} <: AbSignedUniDirBasis{F}
    u::Vector{F}
    sgn::F
end
Softplus{t}(u::Vector{F}, s::F) where {t, F} = Softplus{t, F}(u, s)
Softplus{t}(u::Vector{F}) where {t, F} = Softplus{t, F}(u, one(F))
Softplus(u::Vector{F}, s::F) where {F} = Softplus{1., F}(u, s)
Softplus(u::Vector{F}) where {F} = Softplus{1., F}(u, one(F))
calc(::Type{Softplus{t, F}}, x) where {F, t} = log1plusexp(t*x)/t
derv(::Type{Softplus{t, F}}, x) where {F, t} = sigmoid(t*x)
###############################################################################
