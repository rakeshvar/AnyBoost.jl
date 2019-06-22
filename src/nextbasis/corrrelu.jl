export SolverCorrRelu

struct SolverCorrRelu{T<:Real} <: AbstractNextBasisSolver
    maxiter::Integer
    xtol::T
    ftol::T
end

function SolverCorrRelu(;
    maxiter::Integer=100,
    ftol::T=1e-5,
    xtol::T=1e-5) where T
    SolverCorrRelu{T}(maxiter, xtol, ftol)
end

"""
Iteratively minimize a linearization. Greedy or Frank-Wolfe
"""
function (self::SolverCorrRelu{T})(prob::Problem{L, R, CS},
                   g::Vector{T},
                   uinit::Vector{T}=zeros(sizep(prob))
                   ) where {T<:Real, L, R<:Relu{T}, CS<:L2Constraint}
    X = prob.X
    u = zeros(uinit)
    obj = calc(R, X*u)'*g                                             #X
    for iter in 1:self.maxiter
        active = find(X*u.≥0)
        # printfmtln("{:2d}) {:+6.4f} {} {:4.2f} 𝒜{}/{}",
                    # iter, obj, round.(Int, 100*u), normcs(prob, u), length(active), length(g))
        ui = -X[active, :]'*g[active]
        ui = scaletonorm(prob.Ω, ui)
        unext = (iter*u + 2*ui)/(iter+2)
        objnext = calc(R, X*unext)'*g                                  #X
        if norm(unext-u) ≤ self.xtol || abs(objnext-obj) ≤ self.ftol
            break
        end
        u = unext
        obj = objnext
    end
    u, obj
end
