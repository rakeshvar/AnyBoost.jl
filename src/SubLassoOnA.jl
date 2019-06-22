
@inline lassoonA!(args...) = lassoonA_glmnet!(args...)

################################################################## Helpers
function getβfromC(βs, C)
    p, K = size(βs)
    nprev = norm(βs[:, 1], 1) # = 0.
    for k in 2:K
        ncurr = norm(βs[:, k], 1)
        if C < ncurr
            # @debug format("\t{:6.4f}({}) < {} < {:6.4f}({})", nprev, k-1, C, ncurr, k)
            return (βs[:,k-1]*(ncurr-C) + βs[:,k]*(C-nprev))/(ncurr-nprev)
        end
        nprev = ncurr
    end
    # @debug format("\t{:6.4f}({}) < {}", nprev, K, C)
    return βs[:, end]
end

getdist(::Type{MSELoss}) = Normal()
getdist(::Type{LogitLoss}) = Binomial()

################################################################## GLMNet
using GLMNet

_gety(prob::Problem) = copy(prob.y)
_gety(prob::Problem{LogitLoss}) = [1-prob.y prob.y]

function lassoonA_glmnet!(ens::FWEnsemble, prob::Problem{L}, C) where L
  Z = getZ(ens)
  path = glmnet!(Z, _gety(prob), getdist(L), intercept=false, standardize=false)
  updateβ!(ens, getβfromC(path.betas, C))
end

################################################################## Lasso.jl
# using Lasso
#
# function lassoonA_lassojl(prob::Problem{L}, ens::Ensemble, C) where L
#     lfit = fit(LassoPath, Z, prob.y, getdist(L),
#                 intercept=false, standardize=false)
#     β = getβfromC(lfit.coefs, C)
# end
