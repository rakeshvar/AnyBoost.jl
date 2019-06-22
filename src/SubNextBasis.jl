
defaultsolver(prob::Problem{L, B, CS}) where {L,
                   B<:AbSignedUniDirBasis,
                   CS<:BasisL2Constraint} = SolverLevMarq()

defaultsolver(prob::Problem{L, B, CS}) where {L,
                  B<:Relu,
                  CS<:L2Constraint} = SolverCorrRelu()

function getnextω(prob::Problem{L, B, CS},
                  solvernextω::SolverLevMarq,
                  gr::Vector) where {L,
                                     B<:AbSignedUniDirBasis,
                                     CS<:BasisL2Constraint}
  lsu = prob.X\gr
  upos, ssepos = solvernextω(prob, gr, lsu; useprojections=false, savetrace=false)
  uneg, sseneg = solvernextω(prob, -gr, -lsu; useprojections=false, savetrace=false)
  # printfmt("\tLoss+ = {:.6f} Loss- = {:.6f}", ssepos, sseneg)

  if ssepos ≤ sseneg
    u, s, sse = upos, -1, ssepos      # Sign is flipped because we want to
  else                                # minimize |< gr, ϕ(u)>|
    u, s, sse = uneg, +1, sseneg      # (where as the Least Sq maximizes it)
  end

  # println("Norm ", normcs(prob, u))
  ρ, u = scaleproject(prob.Ω, u)
  # printfmtln("\t Picking {:+2.0f} ρ={:.6f}", s, ρ)
  # println("Norm ", normcs(prob, u))
  B(u, s), ρ, sse
end

function getnextω(prob::Problem{L, B, CS},
                  solvernextω::SolverCorrRelu,
                  gr::Vector) where {L,
                                    B<:Relu,
                                    CS<:L2Constraint}
    lsu = prob.X\gr
    upos, ippos = solvernextω(prob, gr, lsu)
    uneg, ipneg = solvernextω(prob, -gr, -lsu)

    if ippos ≤ ipneg
      u, s, ip = upos, +1, ippos
    else
      u, s, ip = uneg, -1, ipneg
    end

    # @debug format("\tIP+ = {:.6f} IP- = {:.6f} Picking {:+2.0f}", ippos, ipneg, s)
    B(u, s), 1., ip
end
