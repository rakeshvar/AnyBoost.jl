abstract type AbstractNextBasisSolver end

include("nextbasis/levmarq.jl")
include("nextbasis/corrrelu.jl")
include("nextbasis/projgraddesc.jl")

################################################################################
type SolverProxLiner <: AbstractNextBasisSolver
end

################################################################################
type SolverStageWiseGaussNewton <: AbstractNextBasisSolver
end

################################################################################
type SolverGaussNewton <: AbstractNextBasisSolver
end

################################################################################
type SolverStocGradientDescent <: AbstractNextBasisSolver
end

################################################################################
type SolverNLOpt <: AbstractNextBasisSolver
end

################################################################################
