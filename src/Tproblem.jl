export Problem, loss, lossgrad, lossngrad, jacobian, valueandjacobian
export project, normcs, sizen, sizep, predictor, prediction

type Problem{L<:AbLoss,
             B<:AbBasis,
             CS<:AbConstraintSet}
  X :: Matrix
  y :: Vector
  loss :: L
  ϕ :: Type{B}
  Ω :: CS
end

Problem(X, y, l, B, ::Type{BasisL2Constraint}) = Problem(X, y, l, B, BasisL2Constraint(B, X))
Problem(X, y, l, B, ::Type{CS}) where CS<:AbConstraintSet = Problem(X, y, l, B, CS())

################### Pass through sub-sub-routines #############################
loss(prob::Problem, z::Vector) = prob.loss(prob.y, z)
lossgrad(prob::Problem, z::Vector) = derv(prob.loss, prob.y, z)
lossngrad(prob::Problem, z::Vector) = valnderv(prob.loss, prob.y, z)

jacobian(prob::Problem{L,B}, u) where {L,B} = jacobian(B, prob.X, u)
valueandjacobian(prob::Problem{L, B}, u) where {L,B} = valueandjacobian(B, prob.X, u)

project(prob::Problem, u) = project(prob.Ω, u)
normcs(prob::Problem, u) = normcs(prob.Ω, u)

############################### Sub-routines ##################################
### Basics
sizen(prob::Problem) = length(prob.y)
sizep(prob::Problem) = size(prob.X)[2]

### Advanced
predictor(X::Matrix, ω::AbBasis) = calc(ω, X)
predictor(prob::Problem, ω::AbBasis) = predictor(prob.X, ω)
predictor(prob::Problem{L, B}, u::Vector) where {L,B<:AbUniDirBasis}= calc(B, prob.X, u)
prediction(X::Matrix{<:Real}, ωs::Vector{<:AbBasis}, βs::Vector{<:Real}) = sum(βs .* calc.(ωs, (X,)))
prediction(prob::Problem, args...) = prediction(prob.X, args...)

############################### Splitter ##################################
function getsplitter(prob::Problem)
    n, p = size(prob.X)
    medians = [median(prob.X[:, j]) for j in 2:p]
    actives = [prob.X[:, j] .≥ medians[j-1] for j in 2:p]
    u0s = zeros(p, p-1)
    u0s[1, :] = medians
    for j in 2:p
        u0s[j, j-1] = -1.
        u0s[:, j-1] /= normcs(prob, u0s[:, j-1])
    end
    j -> u0s[:, j-1]
end

function getsplits(prob::Problem{L, B}) where {L, B<:AbBasis}
    spliter = getsplitter(prob)
    [B(su*spliter(j), s) for j in 2:sizep(prob) for su in (-1., 1.) for s in (-1., 1.)]
end

function cardinalsplitdirs(prob::Problem{L, B})  where {L,B}
    stdlib = B[]
    p = sizep(prob)
    for j in 2:p
        u = zeros(p)
        u[j] = 1
        u[1] = -median(prob.X[:, j])
        u /= normcs(prob, u)
        for su in (-1., 1.)
            for s in (-1., 1.)
                push!(stdlib, B(su*u, s))
    end end end
    stdlib
end

function randomsplitdir(prob::Problem{L, B})  where {L,B}
    p = sizep(prob)
    u = randn(p)
    u[1] = 0.
    Xu = prob.X*u
    med = median(Xu)
    u[1] = -med
    u /= normcs(prob, u)
    su, s = rand([-1., 1.], 2)
    B(su*u, s)
end

randsplitdirs(prob, num) = [randomsplitdir(prob) for j in 1:num]
