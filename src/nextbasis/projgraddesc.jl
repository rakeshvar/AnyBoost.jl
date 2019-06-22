export SolverProjGradDesc

struct SolverProjGradDesc{T<:Real} <:AbstractNextBasisSolver
    maxiter::Int
    batchsize::Int
    projiter::Int
    stepsize::T
    ftol::T
end

function SolverProjGradDesc(;
    maxiter::Int=100,
    batchsize::Int=20,
    projiter::Int=10,
    stepsize::T=.01,
    ftol::T=1e-4
    ) where T
    SolverProjGradDesc{T}(maxiter, batchsize, projiter, stepsize, ftol)
end

function (self::SolverProjGradDesc{T})(prob::Problem{L, B, CS},
                            r::Vector{T},
                            uinit::Vector{T}) where {L, B<:AbUniDirBasis, CS, T}
    u = copy(uinit)
    n = sizen(prob)
    obj = r'*calc(B, prob.X, u)
    iter = 0

    for i in 1:self.maxiter
        for j in 1:self.batchsize:n
            j2 = min(n, j+self.batchsize-1)
            X = view(prob.X, j:j2, :)
            J = jacobian(B, X, u)
            u -= self.stepsize * J' * r[j:j2]

            # Project
            if (iter .% self.projiter == 0) || (j2 == n)
                u = projectΩ(prob, u)
            end
            iter += 1
        end
        # Stopping
        Δobj = r'*calc(B, prob.X, u) - obj
        obj += Δobj
        abs(Δobj) < self.ftol && break
        printfmtln("\t\t\t obj={:8.4f}", obj)
    end
    u, obj
end
