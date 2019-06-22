using AnyBoost
using AnyBoost.Utils
using Plots
using DataFrames
using CSV
try
    using XGBoost
catch
    println("!!!!!!!!!!!!!!!!!!! XGBoost not installed. Will Error eventually.")
end

const def_params = [
    :eta=>.3,
    :max_depth=>6
    ]
xgloss(::AltStage.MSELoss) = "reg:linear"
xgloss(::AltStage.LogitLoss) = "binary:logistic"

function xgb(trdata, tedata, l; numround=300)
    trdm = DMatrix(trdata.X, label = trdata.y)
    tedm = DMatrix(tedata.X, label = tedata.y)
    bst = xgboost(trdm, numround, objective=xgloss(l))
    al = altloss(l)
    trerrs = Array{Float64}(numround)
    teerrs = Array{Float64}(numround)
    altrerrs = Array{Float64}(numround)
    alteerrs = Array{Float64}(numround)
    norms = Array{Float64}(numround)
    trnpredlast = zeros(trdata.y)
    time = @elapsed for i in 1:numround
        trnpred = MyXGBoost.predict(bst, trdata.X, ntree_limit=i, output_margin=true)  # Get untransformed for logistic
        trerrs[i] = l(trdata.y, trnpred)/trdata.n
        altrerrs[i] = al(trdata.y, trnpred)/trdata.n
        tespred = MyXGBoost.predict(bst, tedata.X, ntree_limit=i, output_margin=true)
        teerrs[i] = l(tedata.y, tespred)/tedata.n
        alteerrs[i] = al(tedata.y, tespred)/tedata.n
        norms[i] = norm(trnpred-trnpredlast)
        trnpredlast = trnpred
    end
    cards = collect(1:numround)
    norms = cumsum(norms)
    bayes = l(tedata.y, tedata.signal)/tedata.n
    albayes = al(tedata.y, tedata.signal)/tedata.n
    cards, norms, trerrs, teerrs, bayes, altrerrs, alteerrs, albayes, time
end
