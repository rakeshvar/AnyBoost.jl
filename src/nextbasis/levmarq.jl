export SolverLevMarq

struct SolverLevMarq{T<:Real} <: AbstractNextBasisSolver
  maxiter::Integer
  xtol::T
  gtol::T
  ftol::T
  λinit::T
  λinc::T
  λdec::T
  min_step_quality::T
  good_step_quality::T
  λmax::T
  λmin::T
  diagmin::T
end

function SolverLevMarq(;
  maxiter::Integer=100,
  xtol::T=1e-8,
  gtol::T=1e-8,
  ftol::T=1e-5,
  λinit::T=1.0,
  λinc::T=10.,
  λdec::T=10.,
  min_step_quality::T=1e-3,
  good_step_quality::T=.75,
  λmax::T = 1e12,
  λmin::T = 1e-12,
  diagmin::T = 1e-6
  ) where T
  @assert 0 ≤ min_step_quality < good_step_quality ≤ 1
  SolverLevMarq{T}(maxiter, xtol, gtol, ftol,
                λinit, λinc, λdec,
                min_step_quality, good_step_quality,
                λmax, λmin, diagmin)
end

function (self::SolverLevMarq{T})(prob::Problem{L, B, CS},
                                  r::Vector{T},
                                  uinit::Vector{T};
                                  useprojections::Bool=true,
                                  savetrace::Bool=false) where {T, L, B, CS}
  u = copy(uinit)
  λ = self.λinit
  xconverged, gconverged, λconverged, fconverged = false, false, false, false
  Δu = zeros(u)
  tr = DataFrame(i=Int[], sse=T[], prsse=T[], trsse=T[], ρ=T[],
                 λ=Int[], lo=String[], ri=String[],
                 normΔu=T[], normg=T[], expdecr=T[]
                 )  # TODO: Time consuming
  local F::Vector{T}, J::Matrix{T}

  for iter in 1:self.maxiter
    F, J = valueandjacobian(prob, u)          # TODO: Time consuming
    F -= r
    sse = sum(abs2, F)
    JJ = J'*J
    D = Diagonal(JJ)                # TODO: Time consuming
    D[0 .< D .< self.diagmin] = self.diagmin  # TODO: -do-
    Jf = -J'*F                      # TODO: transp is Time consuming
                                    # This is also the neg gradient
    normg = norm(Jf, Inf)
    gconverged = normg < self.gtol
    gconverged && break

    accept_step = false
    while !(accept_step || λconverged || fconverged)
      JJD = JJ + λ*D                # TODO: Time consuming
      Δu = JJD\Jf
      if useprojections
        Δu = projectΩ(prob, u+Δu)-u
      end

      # Try this step and compare with linear prediction
      pred_sse = sum(abs2, J*Δu+F)
      trialf = predictor(prob, u+Δu) - r    # TODO: Time consuming   #X
      trial_sse = sum(abs2, trialf)
      expd_decr = 1 - pred_sse/sse          # Positive
      fconverged = expd_decr < self.ftol

      ρ = (sse-trial_sse)/(sse-pred_sse)
      lower_trust = ρ ≤  self.min_step_quality
      raise_trust = ρ > self.good_step_quality

      normΔu = norm(Δu)
      logrelλ = round(Int, log(self.λinc, λ))
      trentry = [iter, sse, pred_sse, trial_sse, ρ,
                 logrelλ,
                 lower_trust ? "↑" : " ",
                 raise_trust ? "↓" : " ",
                 normΔu, normg, expd_decr]
      push!(tr, trentry)

      if lower_trust
        if λ==self.λmax
          λconverged = true
        else
          λ = min(λ*self.λinc, self.λmax)
        end
      else        #Accept this step
        accept_step = true
        u += Δu
        if raise_trust
          λ = max(λ/self.λdec, self.λmin)
        end
      end
    end   #Inner Loop over λ

    xconverged = norm(Δu) < self.xtol
    (xconverged || gconverged || λconverged || fconverged) && break
  end    #Outer Loop over Δu

  savetrace && print(tr)

  u, sum(abs2, predictor(prob, u)-r)  #X
end
