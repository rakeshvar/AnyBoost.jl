using CSV, StatsBase

jp(csv) = joinpath(@__DIR__, csv)

datasets = Dict(
:toy1 => (jp("./data/toy/toy1.csv"), 90, :Y),
:toy2 => (jp("./data/toy/toy2.csv"), 90, :Y),
:wine => (jp("./data/wine/wine5K_f32.csv"), 3600, :quality),
:garnet => (jp("./data/garnet/garnet_norm_f32.csv"), 9000, :TiO2),
)

type RealData{T}
  X::AbstractMatrix{T}
  signal::AbstractVector{T}
  noise::AbstractVector{T}
  y::AbstractVector{T}
  n::Int
end

RealData(X, y) = RealData(X, y, y*0, y, length(y))

loadrealdata(datasetname::Symbol) = loadrealdata(datasets[datasetname]...)

function loadrealdata(csvfilename::String, ntrain::Int, yname::Symbol)
    data = CSV.read(csvfilename)
    disallowmissing!(data)
    ntotal = nrow(data)

    srand(20181103)
    testindices = sample(1:ntotal, ntotal-ntrain, replace=false)
    trainindices = trues(ntotal)
    trainindices[testindices] = false

    test = data[testindices, :]
    train = data[trainindices, :]

    xnames = [name for name in names(data) if name != yname]
    trainx = Matrix(train[:, xnames])
    trainy = Vector(train[yname])
    testx = Matrix(test[:, xnames])
    testy = Vector(test[yname])

    RealData(trainx, trainy), RealData(testx, testy)
end
