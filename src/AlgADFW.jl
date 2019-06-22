
export ADFW, ADFWPath

function ADFWPath(prob::Problem{L, B, CS},
                Cs::Vector{F},
                subsolver::AbstractNextBasisSolver=defaultsolver(prob);
                warmstart=true,
                kwargs...
                ) where {F, L, B<:AbBasis{F}, CS}
    enses = Vector{FWEnsemble{F,B}}()
    for (i, c) in enumerate(Cs)
        println("C = $c")
        ens0 = warmstart && (i > 1) ? enses[end] : FWEnsemble(prob)
        push!(enses, ADFW(prob, c, subsolver; ens=ens0, kwargs...))
    end
    enses
end

function ADFW(prob::Problem{L, B, CS},
              C::F,
              subsolver::AbstractNextBasisSolver=defaultsolver(prob);
              fc=true,                              # Fully Corrective
              ad=false,                             # Alternate Descent
              ens::FWEnsemble=FWEnsemble(prob),     # Warm Start
              maxiter::Int = 300,
              maxjter::Int = 10) where {L, B, CS, F<:AbstractFloat}
  ens = deepcopy(ens)
#  sanitycheck(ens)
  if length(ens) > 0            # warmstart
      @assert fc
      lassoonA!(ens, prob, C)
  end

  iter, jter = 0, 0
  currloss = trainerr(ens)
  # llog(txt="") = printfmtln("{:2d}:{:2d}) C={:5.0f} L={:8.4f} |β|={:.2f}({}) {}", iter, jter, C, currloss, sum(abs, ens.β), countnz(ens.β), txt)
  llog(t="") = nothing
  llog("NULL")

  for iter in 1:maxiter
 #   sanitycheck(ens)
    prevloss = currloss
    objtv, grad = lossngrad(prob, ens.fit)
    ωnew, ρ, score = getnextω(prob, subsolver, grad) # TODO : Get z
    push!(ens, ωnew, 0.)
    # sanitycheck(ens)

    if !fc
        fwvanillaupdate!(ens, iter, C)
  #      sanitycheck(ens)
    else
        lassoonA!(ens, prob, C)
#        sanitycheck(ens)
    end

    currloss = trainerr(ens)
    if ad
        jter = 0
        llog("new (ω, β)")
        if !fc                              # Vanilla FW w/ AD ⇒ One update only
            ens, currloss = backpropωs(prob, ens)
#            sanitycheck(ens)
        else                                # Corrective FW w/ AD ⇒ sub-iterations
            adconverged = false
            while jter < maxjter && !adconverged
                jter += 1
                beforeadloss = currloss
                newens, currloss = backpropωs(prob, ens)
                llog("ad {ω} ∇ωs = $(ssedir(ens, newens))")
                ens = newens
 #               sanitycheck(ens)
                lassoonA!(ens, prob, C)
 #               sanitycheck(ens)
                currloss = trainerr(ens)
                adconverged = beforeadloss-currloss<1e-3
                llog("ad β (converged=$adconverged)")
            end
        end
    end

    llog("END $iter")
    abs(prevloss-currloss) < 1e-5 && break
  end

  llog("\n<--END-->")
  #sanitycheck(ens)
  ens
end
