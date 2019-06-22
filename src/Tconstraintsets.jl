export AbConstraintSet, L1Constraint, L2Constraint, BasisL2Constraint, LInftyConstraint
export project, projectedge, normcs, scaletonorm, scaleproject

abstract type AbConstraintSet end

normcs(cs::CS, u) where CS<:AbConstraintSet = error("Not implemented for $CS")
project(cs::CS, u) where CS<:AbConstraintSet = error("Not implemented for $CS")
projectedge(cs::CS, u) where CS<:AbConstraintSet = error("Not implemented for $CS")
scaletonorm(cs::CS, u::Vector) where CS<:AbConstraintSet = u*cs.radius/normcs(cs, u)

###########################################################################  ℓ1
struct L1Constraint <: AbConstraintSet
  radius :: Real
  fromindex ::Int
end
L1Constraint() = L1Constraint(1., 2)
L1Constraint(r::Real) = L1Constraint(r, 2)

normcs(cs::L1Constraint, u::Vector) = norm(u[cs.fromindex:end], 1)

function project(cs::L1Constraint, u::Vector)
    uproj = projectl1ball(u[cs.fromindex:end], cs.radius)
    vcat(u[1:(cs.fromindex-1)], uproj)
end

function projectedge(cs::L1Constraint, u::Vector)
    uproj = projectl1ballegde(u[cs.fromindex:end], cs.radius)
    vcat(u[1:(cs.fromindex-1)], uproj)
end

function projectsimplex(u0, radius=1)
    u = sort(u0, rev=true)
    cssv = cumsum(u)
    rho = findlast(u.*(1:length(u)) .> (cssv-radius))
    theta = (cssv[rho]-radius)/rho
    max.(u0-theta, 0.)
end

projectl1balledge(u, radius=1) = sign.(u) .* projectsimplex(abs.(u), radius)
projectl1ball(u, radius=1) = sum(abs, u) <= radius ?  u : projectl1balledge(u, radius)

###########################################################################  ℓ2
struct L2Constraint <: AbConstraintSet
  radius :: Real
  fromindex :: Int
end
L2Constraint() = L2Constraint(1., 2)
L2Constraint(r::Real) = L2Constraint(r, 2)

normcs(cs::L2Constraint, u::Vector) = norm(u[cs.fromindex:end], 2)

function project(cs::L2Constraint, u::Vector)
    uproj = u[cs.fromindex:end] * min(1, cs.radius/norm(u[cs.fromindex:end]))
    vcat(u[1:(cs.fromindex-1)], uproj)
end

function projectedge(cs::L2Constraint, u::Vector)
    uproj = u[cs.fromindex:end] * cs.radius/norm(u[cs.fromindex:end])
    vcat(u[1:(cs.fromindex-1)], uproj)
end

###########################################################################  ℓ∞
struct LInftyConstraint <: AbConstraintSet
  radius :: Real
  fromindex :: Int
end
LInftyConstraint() = LInftyConstraint(1., 2)
LInftyConstraint(r::Real) = LInftyConstraint(r, 2)

normcs(cs::LInftyConstraint, u::Vector) = maximum(abs, u[cs.fromindex:end])

function project(cs::LInftyConstraint, u::Vector)
    uproj = clamp.(u[cs.fromindex:end], -cs.radius, cs.radius)
    vcat(u[1:(cs.fromindex-1)], uproj)
end

function projectedge(cs::LInftyConstraint, u::Vector)
    uproj = cs.radius * sign.(u[cs.fromindex:end])
    vcat(u[1:(cs.fromindex-1)], uproj)
end

#####################################################################  Basis ℓ2
struct BasisL2Constraint{B} <: AbConstraintSet
  basis::Type{B}
  X
  radius::Real
end
BasisL2Constraint(::Type{B}, X) where B = BasisL2Constraint{B}(B, X, 1.)

normcs(cs::BasisL2Constraint{B}, u::Vector) where B = norm(calc(B, cs.X, u))

_scaler(::Type{R}, x) where {R<:Relu} = x
_scaler(::Type{R}, x) where {R<:Requ} = √x

function project(cs::BasisL2Constraint{B}, u::Vector) where {B<:AbUniDirBasis}
  u * _scaler(B, min(1, cs.radius/normcs(cs, u)))
end

function projectedge(cs::BasisL2Constraint{B}, u::Vector) where {B<:AbUniDirBasis}
  u * _scaler(B, cs.radius/normcs(cs, u))
end

function scaleproject(cs::BasisL2Constraint{B}, u::Vector) where {B<:AbUniDirBasis}
    scalefactor = cs.radius/normcs(cs, u) #min(1, cs.radius/normcs(cs, u))
    1/scalefactor, u*_scaler(B, scalefactor)
end

#########################################################################  ℓend
