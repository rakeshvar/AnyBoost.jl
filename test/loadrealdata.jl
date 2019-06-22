using CSV, StatsBase

jp(csv) = joinpath(@__DIR__, csv)

datasets = Dict(
"toy1" => (jp("../data/toy/toy1.csv"), 90, :Y),
"toy2" => (jp("../data/toy/toy2.csv"), 90, :Y),
"wine" => (jp("../data/wine/wine5K_f32.csv"), 3600, :quality),
"garnet" => (jp("../data/garnet/garnet_norm_f32.csv"), 9000, :TiO2),
)

loadrealdata(datasetname::String) = loadrealdata(datasets[datasetname]...)

function loadrealdata(csvfilename::String, nuse::Int, yname::Symbol)
    data = CSV.read(csvfilename)
    disallowmissing!(data)
    nall = nrow(data)

    srand(20181103)
    leftout = sample(1:nall, nall-nuse, replace=false)
    leftin = trues(nall)
    leftin[leftout] = false

    dataout = data[leftout, :]
    data = data[leftin, :]

    xnames = [name for name in names(data) if name != yname]
    datax = Matrix(data[:, xnames])
    datay = Vector(data[yname])
    dataoutx = Matrix(dataout[:, xnames])
    dataouty = Vector(dataout[yname])

    datax, datay, dataoutx, dataouty
end
