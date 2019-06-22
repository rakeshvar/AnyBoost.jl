export Stagewise

function Stagewise(prob::Problem{L, B, CS},
                   ϵ::Real,
                   subsolver::AbstractNextBasisSolver=defaultsolver(prob);
                   ad::Bool=false,
                   lasso::Bool=false,
                   tol::Real=1e-8,
                   maxiter::Int=1000) where {L, F, B<:AbBasis{F}, CS}
    ens = SWEnsemble(prob, ϵ, ϵ)
    enses = BBEnsemble{F,B}[]
    iter = 0
    λ = Inf
    currloss = trainerr(ens)
    # llog(txt="") = printfmtln("{:3d}) {:.3f} λ={:.3f} |β|={:.1f}ϵ({}) {}", iter, currloss, λ, sum(abs, ens.β)/ϵ, countnz(ens.β), txt)
    llog(t="") = nothing
    llog()

    # Get first ω
    objtv, grad = lossngrad(prob, ens.fit)
    ω, ρ, score = getnextω(prob, subsolver, grad)
    pushnew!(ens, ω)
    ad && push!(enses, BBEnsemble(ens))
    newloss = trainerr(ens)
    λ = (currloss-newloss)/ϵ
    currloss = newloss
    iter += 1
    llog()

    ϵlosses(d) = [(βi≈0. ? Inf : loss(prob, ens.fit+d*ϵ*z)) for (βi,z) in zip(ens.β, ens.zs)]

    for iter in 2:maxiter
        prevloss = currloss

        # Try decreasing |β|
        dec_losses = ϵlosses(-1)
        declossstar, distar = findmin(dec_losses)
        if declossstar - (lasso ? (λ*ϵ) : tol) ≤ currloss - tol       # W/o tol, guy just in will be out
            decrementβi!(ens, distar)
            currloss = declossstar
            llog("⇩ $distar ! $(ens.β[distar])")
        else

        # Get a new predictor
          objtv, grad = lossngrad(prob, ens.fit)
          ω, ρ, score = getnextω(prob, subsolver, grad)
          z = predictor(prob, ω)
          newzloss = loss(prob, ens.fit + ϵ*z)

        # Try incrementing old ones
          inc_losses = ϵlosses(1)
          inclossstar, instar = findmin(inc_losses)

          if currloss < min(inclossstar, newzloss)
              printfmtln("Can not continue at {:d} c={:.4f} < (d={:.4f}({}), i={:.4f}({}), n={:.4f}).\n Try decreasing ϵ!",
                iter, currloss, declossstar, distar, inclossstar, instar, newzloss)
              break
          end

          if inclossstar ≤ newzloss
            augmentβi!(ens, instar)
            currloss = inclossstar
            # llog("⇑ $instar !")
          else
            pushnew!(ens, ω, z)
            currloss = newzloss
            llog("new (ω, β)!")
          end
          λ = min(λ, (prevloss-currloss-tol)/ϵ)
        end

        if ad
            ens, currloss = backpropωs(prob, ens)
            push!(enses, BBEnsemble(ens))
            llog("ad {ω}")
        end
    end

    llog("\nEND")
    ad ? enses : ens
end
