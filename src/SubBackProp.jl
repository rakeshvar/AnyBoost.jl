
jacobian(::Type{B},
         X::AbstractMatrix{F},
         u::Vector{F}) where {B<:AbUniDirBasis, F<:Real} = derv(B, X*u) .* X

jacobian(ω::B, X) where {B<:AbUniDirBasis} = derv(ω, X) .* X

function valueandjacobian(::Type{B},
                    X::Matrix{F},
                    u::Vector{F}) where {B<:AbUniDirBasis, F<:Real}
  Xu=X*u
  calc(B, Xu), derv(B, Xu).*X
end

function backpropωs(prob::Problem{L, B, CS}, ens::AbEnsemble{F, B};
    maxiter::Int=10, αbtls::F=.3, βbtls::F=.25, maxjter::Int=9, ftol::F=1e-6) where {F, L, B, CS}

    m = length(ens)
    p = sizep(prob)
    ∇L_u = Array{F}(p, m)
    currloss = loss(prob, ens.fit)
    converged = false
    # printfmt("    0: 0) L={:6.4f}", currloss)

    t = 1/sizen(prob)
    for i in 1:maxiter
        ∇L_z = derv(L(), prob.y, ens.fit)
        for j in 1:m
            ∇L_u[:, j] = ens.β[j] * jacobian(ens.ωs[j], prob.X)' * ∇L_z
        end

        normsqgrad = sum(abs2.(∇L_u))
        # printfmtln("\t\t|∇f|={:8.6f}", normsqgrad)
        jter = 0
        while jter < maxjter          #Back-tracking Line Search
            jter += 1
            newens = deepcopy(ens)
            stepalong!(newens, ∇L_u, t)
            steplnsq = ssedir(ens, newens)
            newloss = loss(prob, newens.fit)
            # printfmt("   {:2d}:{:2d}) L={:6.4f} t={:.6f}", i, jter, newloss, t)
            if newloss < currloss - αbtls*t*steplnsq
                percentgain = 1 - newloss/currloss
                # printfmtln("\t\tBTLS success %gain={:9.7f}", percentgain)
                converged = percentgain < ftol
                currloss = newloss
                ens = newens
                break
            end
            t *= βbtls
        end
        if normsqgrad < 1e-6
            # printfmtln("   {:2d}) norm too small", i)
            break
        end
        if jter == maxjter
            # printfmtln("   {:2d}) t too small", i)
            break
        end
        if converged
            # printfmtln("   {:2d}) Converged", i)
            break
        end
    end
    # println("\t\tNorms: ", round.(Int, 100*[normcs(prob, ens.ωs[j].u) for j in 1:m]))
    ens, currloss
end
