export NeuralNetwork, NNPath

# const NNEnsemble{F, B} = FWEnsemble{F, B}

function _randominit(prob::Problem{L, B, CS}, C::F, nnodes::Int)  where {L, B<:AbSignedUniDirBasis, CS, F<:AbstractFloat}
    ens = FWEnsemble(prob)
    β = ones(nnodes)*√C/nnodes #projectl1ball(randn(nnodes), C/2.)
    for j in 1:nnodes
        u = project(prob, randn(sizep(prob)))
        sgn = rand([-1, 1])
        push!(ens, B(u, sgn), 0.*β[j])
    end
    ens
end

NNPath(prob::Problem{L, B, CS}, Cs::Vector{F}, nnodes::Int;
       kwargs...) where {F, L, B<:AbBasis{F}, CS} =
        [NeuralNetwork(prob, c, nnodes; kwargs...) for c in Cs]

function NeuralNetwork(prob::Problem{L, B, CS},
              C::F,
              nnodesorens::Union{Int,FWEnsemble};
              stepsize=1e-2, βstep=:safe,
              maxiter=5000) where {L, B, CS, F<:AbstractFloat}
    if isa(nnodesorens, Int)
        nnodes = nnodesorens
        ens = _randominit(prob, C, nnodes)
    else
        ens = deepcopy(nnodesorens)
        nnodes = length(ens)
    end
    ∇L_u = Array{F}(sizep(prob), nnodes)
    ∇L_β = similar(ens.β)
    currloss = trainerr(ens)
    iter = 0
    # llog(txt="") = printfmtln("{:3d}) {:6.6f} {:.2f}({}) {}", iter, currloss, sum(abs, ens.β), countnz(ens.β), txt)
    llog(t="") = nothing
    llog("Initial")
    errs = fill(NaN, 2maxiter+1)
    errs[1] = currloss
    lim = NaN
    projected = false
    σ1sqrd = NaN
    if βstep === :safe
        z=getZ(ens)
        ZZ=z'*z
        σ1sqrd = eig(ZZ)[1][end]
    elseif βstep === :limited
        lim = 16C/maxiter/nnodes
    end

    for iter in 1:maxiter
        prevloss = currloss
        step = stepsize/√iter
        ∇L_z = derv(L(), prob.y, ens.fit)
        for j in 1:nnodes
            ∇L_u[:, j] = ens.β[j] * jacobian(ens.ωs[j], prob.X)' * ∇L_z
        end
        stepalong!(ens, ∇L_u, step)
        # sanitycheck(ens)
        currloss = trainerr(ens)
        llog("new {ω}")
        errs[2iter] = currloss
        if βstep === :lasso
            Z = getZ(ens)
            path = glmnet!(Z, _gety(prob), getdist(L), intercept=false, standardize=false)
            β = getβfromC(path.betas, C)
        else
            ∇L_β = [ens.zs[j]' * ∇L_z for j in 1:nnodes]
            if βstep === :safe
                Δβ = -∇L_β/σ1sqrd
            elseif βstep === :verysafe
                z=getZ(ens);  ZZ=z'*z;  σ1sqrd = eig(ZZ)[1][end]
                Δβ = -∇L_β/σ1sqrd
            elseif βstep === :newton
                z=getZ(ens);  ZZ=z'*z
                Δβ = -ZZ\∇L_β
            elseif βstep === :limited
                ∇L_β /= maximum(abs, ∇L_β) / lim
                Δβ = -∇L_β
                println(sum(Δβ.*ens.β .> 0))
            elseif βstep===:none
                Δβ = -step*∇L_β
            else
                error("βstep not understood: ", βstep)
            end
            β = ens.β + Δβ
            projected = sum(abs, β) > C
            false && println(round.(Int, 1*β), sum(abs, β))
            β = projectl1ball(β, C)
            false && println(round.(Int, 1*β), sum(abs, β))
        end
        updateβ!(ens, β, false)
        # sanitycheck(ens)
        currloss = trainerr(ens)
        llog(format("new β |{}| |{:.1f}| {}", countnz(β), sum(abs, β), projected?"!":""))
        errs[2iter+1] = currloss
        (βstep===:lasso) && (abs(1-currloss/prevloss) < 1e-5) && (println("loss converged"); break)
    end
    ens #, errs
end
