export GradientBoost

function GradientBoost(prob::Problem{L, B, CS},
                       rate::Real,
                       subsolver::AbstractNextBasisSolver=defaultsolver(prob);
                       ad=false,
                       maxiter::Int=300) where {L, F, B<:AbBasis{F}, CS}
    ens = GBEnsemble(prob)
    enses = BBEnsemble{F,B}[]
    iter = 0
    currloss = trainerr(ens)
    # llog(txt="") = printfmtln("{:3d}) {:.6f} {:.2f}({}) {}", iter, currloss, sum(abs, ens.β), countnz(ens.β), txt)
    llog(t="") = nothing
    llog()

    for iter in 1:maxiter
        prevloss = currloss
        objtv, grad = lossngrad(prob, ens.fit)
        ω, ρ, score = getnextω(prob, subsolver, grad)
        βi = linesearch(prob, ens.fit, ω)
        βi *= rate
        push!(ens, ω, βi)
        currloss = trainerr(ens)

        if ad
            ens, currloss = backpropωs(prob, ens)
            push!(enses, BBEnsemble(ens))
            llog("ad {ω}")
        end

        abs(1-currloss/prevloss) < 1e-5 && (println("loss converged"); break)
        abs(ens.β[end]) < abs(ens.β[1])*1e-4 && (println("β converged"); break)
        llog(format("{:.2f}%", 100*abs(ens.β[end]) / abs(ens.β[1])))
    end

    llog("\nEND")
    ad ? enses : ens
end
