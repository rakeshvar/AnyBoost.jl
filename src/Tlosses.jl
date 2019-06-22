export AbLoss, MSE, MAD, LOGIT, altloss

abstract type AbLoss end
# Notation
#       y = Observed target
#       z = fit Σ_j ϕ(X, ω_j)
# Methods
#       Loss evaluation
#       Generalized Residual
valnderv(l::AbLoss, y, z) = l(y, z), derv(l, y, z)
altloss(l::AbLoss) =  (_...) -> NaN

################################################################################
type MSELoss <: AbLoss
end

const MSE = MSELoss()

(l::MSELoss)(y, z) = sum(abs2, z-y)/2
derv(::MSELoss, y, z) = z-y
derv2(::MSELoss, y, z) = 1.
function valnderv(::MSELoss, y, z)
  diff = z-y
  sum(abs2, diff)/2, diff
end

################################################################################
type MADLoss <: AbLoss
end

const MAD = MADLoss()

(l::MADLoss)(y, z) =  sum(abs, z-y)
derv(::MADLoss, y, z) = sign.(z-y)
derv2(::MADLoss, y, z) = 0.
function valnderv(::MADLoss, y, z)
    diff = z-y
    sum(abs, diff), sign.(diff)
end

################################################################################
type LogitLoss <: AbLoss
end

const LOGIT = LogitLoss()

log1plusexp(x) = (x<0.) ? log1p(exp(x)) : x+log1p(exp(-x))
sigmoid(x) = (x<0.) ? exp(x)/(1.+exp(x)) : 1./(1.+exp(-x))
(l::LogitLoss)(y, z) =  sum(log1plusexp.(z)) - y'*z
derv(::LogitLoss, y, z) = sigmoid.(z) - y
derv2(::LogitLoss, y, z) = (s=sigmoid.(z); (1-s).*s)
"""
bits(sigmoid(-64)) = "0011101000101001011010011101010001110011001000011110010011001100"
bits(1-(1-sigmoid(-64))) = "0000000000000000000000000000000000000000000000000000000000000000"
"""

misclassification(y, z, threshold=.5) = sum(y .!= (sigmoid.(z) .≥ threshold))
altloss(::LogitLoss) =  misclassification
################################################################################
