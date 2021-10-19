cd("/User/homes/bahrens/Science_guided_ML_chapter_Q10")

# load Project.toml and install packages
using Pkg; Pkg.activate("."); Pkg.instantiate()

using DiffEqFlux, Optim, DiffEqSensitivity, DiffEqOperators, Flux, LinearAlgebra, Interpolations, Plots, Dates, DataFrames, DataFramesMeta, CSV

file = "Synthetic4BookChap.csv"

dfall = CSV.read(file, DataFrame, normalizenames=true, missingstring = "NA", dateformat="yyyy-mm-ddTHH:MM:SSZ")

dfall

println(names(dfall))

dfall = @linq dfall |>
  transform(yearmonth = DateTime.(Year.(:DateTime),Month.(:DateTime))) |>
  transform(yearmonthday = DateTime.(Year.(:DateTime),Month.(:DateTime),Day.(:DateTime))) |>
  transform(yearmonthday = DateTime.(Year.(:DateTime),Month.(:DateTime),Day.(:DateTime))) |>
  transform(yearmonthdayhour = DateTime.(Year.(:DateTime),Month.(:DateTime),Day.(:DateTime),Hour.(:DateTime)))



#===============================================================================
## plotting
===============================================================================#
#Plots.scalefontsizes(1.5)
#gr(size=(1100,700))
#plot(dfall.DateTime,dfall.wdefCum)

#for i in names(dfall)
#    p1 = plot(dfall.DateTime,dfall[:,i], title = i, label = "")
#    png(p1, joinpath("png", i))
#end

#===============================================================================
# # make it work with sciml_train
===============================================================================#
# https://github.com/SciML/DiffEqFlux.jl/issues/432
# for sciml_train

# implement minibatches https://diffeqflux.sciml.ai/v1.24/examples/minibatch/

# https://github.com/SciML/DiffEqFlux.jl/issues/432

using StatsBase

dropmissing!(dfall)

x_dfk = @linq dfall |> 
  select(
    #:TA,
    :SW_POT_sm, :SW_POT_sm_diff
  )
x_Rb_woTA = Array(x_dfk)'

# normalization
#sdf = fit(UnitRangeTransform, x_Rb_woTA, dims=2)
#x_Rb_woTA = Float32.(StatsBase.transform(sdf, x_Rb_woTA) .*2 .- 1)

# standardisation
x_Rb_woTA = Float32.(Flux.normalise(x_Rb_woTA))

plot(x_Rb_woTA')

x_dfk = @linq dfall |> 
  select(
    :TA,
    :SW_POT_sm, :SW_POT_sm_diff
  )
x_Rb_wiTA = Array(x_dfk)'

# normalization
#sdf = fit(UnitRangeTransform, x_Rb_wiTA, dims=2)
#x_Rb_wiTA = Float32.(StatsBase.transform(sdf, x_Rb_wiTA) .*2 .- 1)


x_Rb_wiTA = Float32.(Flux.normalise(x_Rb_wiTA))

plot(x_Rb_wiTA')

# example run with random NN
#NN = FastChain(FastDense(size(x_Rb_wiTA,1), 16, relu),FastDense(16, 16, relu),FastDense(16, 1,x->x^2))
NN = FastChain(FastDense(size(x_Rb_wiTA,1), 5, tanh),FastDense(5, 5, tanh),FastDense(5, 5, tanh),FastDense(5, 5, tanh),FastDense(5, 1,x->x^2))
#NN = FastChain(FastDense(size(x_Rb_wiTA,1), 16, leakyrelu),FastDense(16, 16, leakyrelu),FastDense(16, 1,x->x^2))
p_NN_random = Float32.(initial_params(NN))

Rb_NN_random = NN(x_Rb_wiTA,p_NN_random)

using Statistics
var(Rb_NN_random)

plot(Rb_NN_random')

function fR(Temp,x_Rb;p)
    Q10 = p[1]
    p_NN = p[2:end]
    
    Rb = NN(x_Rb,p_NN)'

    Temp_ref = Float32(15.0) # reference temperature
    
    if(Q10 >= 0.f0)
      R = Rb .* Q10.^((Temp .- Temp_ref) ./ Float32(10.0))
    else 
      R = Rb .* 0.f0
    end
end

p_ini = [Float32(1.0);p_NN_random]

Reco_random = fR(Float32.(dfall.TA),x_Rb_wiTA;p = p_ini)

plot(dfall.DateTime,dfall.RECO_syn)
plot!(dfall.DateTime,Reco_random)

L2(x) = sum(abs2, x)/length(x)

θ = p_ini
x_Rb = x_Rb_wiTA
function loss_simple(θ)
    mod = fR(Float32.(dfall.TA),x_Rb;p = θ)
    cost = Flux.mse(mod,Float32.(dfall.RECO_syn))
end
loss_simple(p_ini)


function loss_adjoint(θ, obs, input, TA)
  mod = fR(TA',input;p = θ)
  cost = Flux.mse(mod',obs)
  return cost
end

using GalacticOptim
optfun = OptimizationFunction((θ, p, obs, input, TA) -> loss_adjoint(θ, obs, input, TA),GalacticOptim.AutoZygote())
optprob = OptimizationProblem(optfun, p_ini)

k = 32 #length(dfall.RECO_syn) #32 #Int(round(length(dfall.RECO_syn)/2.0,digits = 0))
iteratations_in_epoch = round((length(dfall.RECO_syn)/k);digits = 0)

train_loader = Flux.Data.DataLoader((Array(Float32.(dfall.RECO_syn)'),Array(x_Rb_wiTA), Array(Float32.(dfall.TA)')), batchsize = k, partial=false)
using IterTools: ncycle

using MLDataUtils

slide1 = slidingwindow(i->i+2,(Array(Float32.(dfall.RECO_syn)'),Array(x_Rb_wiTA), Array(Float32.(dfall.TA)')),2;stride = 1)
train_loader2 = batchview((Array(Float32.(dfall.RECO_syn)'),Array(x_Rb_wiTA), Array(Float32.(dfall.TA)')),k)

collect(ncycle([3,7], 3))

length_row = length(dfall.RECO_syn)
chunksize = Int(round(length_row/k,digits = 0))

ch1 = Flux.chunk(Float32.(dfall.RECO_syn)', chunksize)
ch2 = Flux.chunk(x_Rb_wiTA, chunksize)
ch3 = Flux.chunk(Float32.(dfall.TA)', chunksize)


typeof(train_loader)

obs = train_loader.data[1]
input = train_loader.data[2]
TA = train_loader.data[3]
l1 = loss_adjoint(p_ini, train_loader.data[1], train_loader.data[2],train_loader.data[3])[1]

function loss_L2(θ)
  mod = fR(Float32.(dfall.TA),x_Rb;p = θ)
  Flux.mse(mod,Float32.(dfall.RECO_syn)) + L2(θ[2:end])
end
loss_L2(p_ini)

using Flux

n_ini = 7
maxtime = 10.0*60.0
function test_training(Q10_ini,opt,maxiters = 100000,maxtime = maxtime, n_ini = n_ini)
  Q10_array = Array{Union{Missing,Float32}}(missings(maxiters+2,n_ini))
  loss_array = Array{Union{Missing,Float32}}(missings(maxiters+2,n_ini))
  var_rb_array = Array{Union{Missing,Float32}}(missings(maxiters+2,n_ini))
  mean_rb_array = Array{Union{Missing,Float32}}(missings(maxiters+2,n_ini))
  time_array = Array{Union{Missing,Float32}}(missings(maxiters+2,n_ini))

  Threads.@threads for i in 1:n_ini
    println("$(i) of $(n_ini)")
    losses = []
    Q10s = []
    var_rb = []
    mean_rb = []
    time_seconds = []
    callback(θ,l) = begin
      push!(losses, l)
      push!(Q10s,θ[1])
      push!(var_rb,Statistics.var(NN(x_Rb,θ[2:end])'))
      push!(mean_rb,Statistics.mean(NN(x_Rb,θ[2:end])'))
      push!(time_seconds,time_ns())
      if length(losses)%100==0
          println("After $(length(losses)) iterations: loss $(losses[end]) | Q10 $(Q10s[end])")
          #println("Current Q10 after $(length(losses)) iterations: $(Q10s[end])")
      end
      if (time_seconds[end] - time_seconds[1])/1e9 > maxtime || length(losses) >= maxiters
        true
      else
        false
      end
    end

    p_NN_random = Float32.(Flux.glorot_uniform(length(initial_params(NN))))
    p_ini = [Float32(Q10_ini);p_NN_random]

    #hybrid_opti_ADAM = DiffEqFlux.sciml_train(loss_simple, p_ini, opt, maxiters = maxiters, cb = callback)
    optprob = OptimizationProblem(optfun, p_ini) # lb = fill(-3.f0, length(p_ini)), ub = fill(3.f0, length(p_ini))
    hybrid_opti_ADAM = GalacticOptim.solve(optprob, opt, ncycle(train_loader2, maxiters); maxiters = maxiters, cb = callback)
    #hybrid_opti_ADAM = DiffEqFlux.sciml_train(loss_adjoint, p_ini, opt, cb = callback, maxiters = maxiters, ncycle(train_loader, maxiters))
    #hybrid_opti_ADAM = DiffEqFlux.sciml_train(loss_simple, p_ini, opt, cb = callback, maxiters = maxiters)
    #hybrid_opti = DiffEqFlux.sciml_train(loss_simple, p_ini, BFGS(initial_stepnorm = 0.001), maxiters = maxiters, cb = callback, allow_f_increases = true)
    Q10_array[1:length(Q10s),i] = Q10s
    loss_array[1:length(losses),i] = losses
    var_rb_array[1:length(var_rb),i] = var_rb
    mean_rb_array[1:length(mean_rb),i] = mean_rb
    time_array[1:length(time_seconds),i] = (time_seconds .- time_seconds[1])/1e9
  end
  (Q10=Q10_array,loss=loss_array,mean_rb = mean_rb_array,var_rb=var_rb_array,time_seconds = time_array)
end


# with TA in NN
#opt = Flux.Optimiser(Flux.ExpDecay(0.5, 0.1, 2000, 1e-4), ADAM())
#opt = Descent(0.01)
#opt = Momentum() # good
#opt = RMSProp() # bad
opt = ADAMW(0.01)
#opt = BFGS(initial_stepnorm = 0.001)
#opt = SimulatedAnnealing()
using BlackBoxOptim
#opt = BBO()
#using CMAEvolutionStrategy
x_Rb = x_Rb_wiTA
NN = FastChain(FastDense(size(x_Rb,1), 16, tanh),FastDense(16, 16, tanh),FastDense(16, 1,x->x^2))
Q10_05 = test_training(0.5,opt)
Q10_15 = test_training(1.5,opt)
Q10_25 = test_training(2.5,opt)

gr(size=(1100,700))
#Plots.scalefontsizes(1.5)
df_loss = DataFrame(Q10_05.loss,:auto)
number_epochs = nrow(dropmissing(df_loss))

p1 = plot(Q10_05.time_seconds,Q10_05.Q10, colour = "blue", labels = "", ylab = "Q₁₀", title = "Number of Epochs $number_epochs")
plot!(Q10_15.time_seconds,Q10_15.Q10, colour = "red", labels = "")
plot!(Q10_25.time_seconds,Q10_25.Q10, colour = "violet", labels = "")




p1b = plot(Q10_05.time_seconds,Q10_05.mean_rb, colour = "blue", labels = "", ylab = "Mean(Rb)")
plot!(Q10_15.time_seconds,Q10_15.mean_rb, colour = "red", labels = "")
plot!(Q10_25.time_seconds,Q10_25.mean_rb, colour = "violet", labels = "")

p2 = plot(Q10_05.time_seconds,Q10_05.var_rb, colour = "blue", labels = "", ylab = "Variance(Rb)")
plot!(Q10_15.time_seconds,Q10_15.var_rb, colour = "red", labels = "")
plot!(Q10_25.time_seconds,Q10_25.var_rb, colour = "violet", labels = "")

p3 = plot(Q10_05.time_seconds,Q10_05.loss, colour = "blue", labels = "", ylab = "loss")
plot!(Q10_15.time_seconds,Q10_15.loss, colour = "red", labels = "")
plot!(Q10_25.time_seconds,Q10_25.loss, colour = "violet", labels = "")

gr(size=(1100,700))
using Plots.PlotMeasures: mm

plot(p1,p1b,p2,p3, layout = (4,1), left_margin=5mm, bottom_margin=5mm, xlab = "Time (seconds)")
xlims!(0,maxtime)


# Now this
# https://sebastiancallh.github.io/post/langevin/
# and this
# https://diffeqflux.sciml.ai/stable/examples/BayesianNODE_SGLD/


scatter(Q10_05.mean_rb,Q10_05.var_rb,Q10_05.Q10,colour = "blue", labels = "")
scatter!(Q10_15.mean_rb,Q10_15.var_rb,Q10_15.Q10,colour = "red", labels = "")
scatter!(Q10_25.mean_rb,Q10_25.var_rb,Q10_25.Q10,colour = "violet", labels = "")

scatter(Q10_25.var_rb[:,2],Q10_25.Q10[:,2])

scatter(collect(skipmissing(Q10_05.loss))[end-n_ini:end],collect(skipmissing(Q10_05.Q10))[end-n_ini:end], label = "0.5", legend = :best, xlab = "loss", ylab = "Q10", left_margin=5mm, bottom_margin=5mm,right_margin=5mm)
scatter!(collect(skipmissing(Q10_15.loss))[end-n_ini:end],collect(skipmissing(Q10_15.Q10))[end-n_ini:end], label = "1.5")
scatter!(collect(skipmissing(Q10_25.loss))[end-n_ini:end],collect(skipmissing(Q10_25.Q10))[end-n_ini:end], label = "2.5")
xlims!(0,0.006)
ylims!(1.3,1.7)

scatter(collect(skipmissing(Q10_05.var_rb))[end-n_ini:end],collect(skipmissing(Q10_05.Q10))[end-n_ini:end], label = "0.5", legend = :topright, xlab = "Variance(Rb)", ylab = "Q10", left_margin=5mm, bottom_margin=5mm,right_margin=5mm)
scatter!(collect(skipmissing(Q10_15.var_rb))[end-n_ini:end],collect(skipmissing(Q10_15.Q10))[end-n_ini:end], label = "1.5")
scatter!(collect(skipmissing(Q10_25.var_rb))[end-n_ini:end],collect(skipmissing(Q10_25.Q10))[end-n_ini:end], label = "2.5")

scatter(collect(skipmissing(Q10_05.mean_rb))[end-n_ini:end],collect(skipmissing(Q10_05.Q10))[end-n_ini:end], label = "0.5", legend = :bottomright, xlab = "Mean(Rb)", ylab = "Q10", left_margin=5mm, bottom_margin=5mm,right_margin=5mm)
scatter!(collect(skipmissing(Q10_15.mean_rb))[end-n_ini:end],collect(skipmissing(Q10_15.Q10))[end-n_ini:end], label = "1.5")
scatter!(collect(skipmissing(Q10_25.mean_rb))[end-n_ini:end],collect(skipmissing(Q10_25.Q10))[end-n_ini:end], label = "2.5")




# without TA in NN
x_Rb = x_Rb_woTA
NN = FastChain(FastDense(size(x_Rb,1), 16, relu),FastDense(16, 16, relu),FastDense(16, 16, relu),FastDense(16, 1, x -> x^2))
Q10_05 = test_training(0.5)
Q10_15 = test_training(1.5)
Q10_25 = test_training(2.5)

function endpoint(x, colour) 
  for i in 1:3
    thepch = (:circle, :utriangle, :dtriangle)
    forthepoint = collect(skipmissing(x[:,i]))
    scatter!([length(forthepoint)],[forthepoint[end]],colour = colour, markershape=thepch[i], labels = "")
  end
end

gr(size=(1100,700))
#Plots.scalefontsizes(1.5)
p1 = plot(Q10_05.time_seconds,Q10_05.Q10, colour = "blue", labels = "", title = "wo TA in NN", ylab = "Q₁₀")
plot!(Q10_15.time_seconds,Q10_15.Q10, colour = "red", labels = "")
plot!(Q10_25.time_seconds,Q10_25.Q10, colour = "violet", labels = "")

p2 = plot(Q10_05.time_seconds,Q10_05.var_rb, colour = "blue", labels = "", title = "wo TA in NN", ylab = "Variance(Rb)")
plot!(Q10_15.time_seconds,Q10_15.var_rb, colour = "red", labels = "")
plot!(Q10_25.time_seconds,Q10_25.var_rb, colour = "violet", labels = "")

p3 = plot(Q10_05.time_seconds,Q10_05.loss, colour = "blue", labels = "", title = "wo TA in NN", ylab = "loss")
plot!(Q10_15.time_seconds,Q10_15.loss, colour = "red", labels = "")
plot!(Q10_25.time_seconds,Q10_25.loss, colour = "violet", labels = "")

gr(size=(1100,700))
using Plots.PlotMeasures: mm
plot(p1,p2,p3, layout = (3,1), left_margin=5mm, bottom_margin=5mm)

# definition L2 training
function test_training_L2(Q10_ini,maxiters = 50,n_ini = 3)
  Q10_array = Array{Union{Missing,Float32}}(missings(maxiters+1,n_ini))
  for i in 1:n_ini
    println("$(i) of $(n_ini)")
    losses = []
    Q10s = []
    #times = []
    callback(θ,l) = begin
      push!(losses, l)
      push!(Q10s,θ[1])
      if length(losses)%3==0
          println("After $(length(losses)) iterations: loss $(losses[end]) | Q10 $(Q10s[end])")
          #println("Current Q10 after $(length(losses)) iterations: $(Q10s[end])")
      end
      false
    end
    p_NN_random = Float32.(Flux.glorot_uniform(length(initial_params(NN))))
    p_ini = [Float32(Q10_ini);p_NN_random]
    hybrid_opti = DiffEqFlux.sciml_train(loss_L2, p_ini, BFGS(initial_stepnorm = 0.01), maxiters = maxiters, cb = callback, allow_f_increases = true)
    Q10_array[1:length(Q10s),i] = Q10s
  end
  Q10_array
end

x_Rb = x_Rb_wiTA
NN = FastChain(FastDense(size(x_Rb,1), 16, relu),FastDense(16, 16, relu),FastDense(16, 1, x -> x^2))
Q10_05_wTA_L2 = test_training_L2(0.5)
Q10_15_wTA_L2 = test_training_L2(1.5)
Q10_25_wTA_L2 = test_training_L2(2.5)

p3 = plot(Q10_05_wTA_L2, colour = "blue", labels = "", title = "wi TA in NN, L2", xlab = "# iterations")
plot!(Q10_15_wTA_L2, colour = "red", labels = "")
plot!(Q10_25_wTA_L2, colour = "violet", labels = "")

gr(size=(1100,700))
using Plots.PlotMeasures: mm
plot(p1,p2,p3, link=:both, width = 1, layout = (1,3), ylab = "Q10",xlab = "iterations", left_margin=5mm, bottom_margin=5mm)


Reco_opti = fR(dfall.TA,x_Rb;p = hybrid_opti.minimizer)

plot(dfall.DateTime,dfall.RECO_syn)
plot!(dfall.DateTime,Reco_opti)

plot(Q10s)
plot(losses)