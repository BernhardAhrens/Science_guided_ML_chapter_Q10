cd("C:/Users/bahrens/OneDrive/Projects/POCMOCML/book_chapter")

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
#Plots.scalefontsizes(1/2)
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

dropmissing!(dfall)

x_dfk = @linq dfall |> 
  select(
    #:TA,
    :SW_POT_sm, :SW_POT_sm_diff
  )
x_Rb_woTA = Array(x_dfk)'

x_Rb_woTA = Float32.(Flux.normalise(x_Rb_woTA))

x_dfk = @linq dfall |> 
  select(
    :TA,
    :SW_POT_sm, :SW_POT_sm_diff
  )
x_Rb_wiTA = Array(x_dfk)'
x_Rb_wiTA = Float32.(Flux.normalise(x_Rb_wiTA))


NN = FastChain(FastDense(size(x_Rb_wiTA,1), 16, tanh),FastDense(16, 16, tanh),FastDense(16, 1,x->x^2))
p_NN_random = Float32.(initial_params(NN))

Rb_NN_random = NN(x_Rb_woTA,p_NN_random)

plot(Rb_NN_random')

function fR(Temp,x_Rb;p)
    Q10 = p[1]
    p_NN = p[2:end]
    
    Rb = NN(x_Rb,p_NN)'

    Temp_ref = Float32(15.0) # reference temperature
    
    R = Rb .* Q10.^((Temp .- Temp_ref) ./ Float32(10.0))
end

p_ini = [Float32(1.0);p_NN_random]

Reco_random = fR(Float32.(dfall.TA),x_Rb_woTA;p = p_ini)

plot(dfall.DateTime,dfall.RECO_syn)
plot!(dfall.DateTime,Reco_random)

L2(x) = sum(abs2, x)/length(x)

θ = p_ini
x_Rb = x_Rb_woTA
function loss_simple(θ)
    mod = fR(Float32.(dfall.TA),x_Rb;p = θ)
    cost = Flux.mse(mod,Float32.(dfall.RECO_syn))
end
loss_simple(p_ini)

function loss_L2(θ)
  mod = fR(Float32.(dfall.TA),x_Rb;p = θ)
  Flux.mse(mod,Float32.(dfall.RECO_syn)) + L2(θ[2:end])
end
loss_L2(p_ini,x_Rb_woTA)

function test_training(Q10_ini,maxiters = 200,n_ini = 3)
  Q10_array = Array{Union{Missing,Float32}}(missings(maxiters+1,n_ini))
  for i in 1:n_ini
    println("$(i) of $(n_ini)")
    losses = []
    Q10s = []
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

    hybrid_opti = DiffEqFlux.sciml_train(loss_simple, p_ini, BFGS(initial_stepnorm = 0.001), maxiters = maxiters, cb = callback)
    Q10_array[1:length(Q10s),i] = Q10s
  end
  Q10_array
end

x_Rb = x_Rb_woTA
NN = FastChain(FastDense(size(x_Rb,1), 16, tanh),FastDense(16, 16, tanh),FastDense(16, 1,x->x^2))
Q10_05 = test_training(0.5)
Q10_15 = test_training(1.5)
Q10_25 = test_training(2.5)

gr(size=(1100,700))
#Plots.scalefontsizes(1.5)
p1 = plot(Q10_05, colour = "blue", labels = "", title = "wo TA in NN")
plot!(Q10_15, colour = "red", labels = "")
plot!(Q10_25, colour = "violet", labels = "")

x_Rb = x_Rb_wiTA
NN = FastChain(FastDense(size(x_Rb,1), 16, tanh),FastDense(16, 16, tanh),FastDense(16, 1,x->x^2))
Q10_05_wTA = test_training(0.5)
Q10_15_wTA = test_training(1.5)
Q10_25_wTA = test_training(2.5)

p2 = plot(Q10_05_wTA, colour = "blue", labels = "", title = "wi TA in NN")
plot!(Q10_15_wTA, colour = "red", labels = "")
plot!(Q10_25_wTA, colour = "violet", labels = "")


function test_training_L2(Q10_ini,maxiters = 200,n_ini = 3)
  Q10_array = Array{Union{Missing,Float32}}(missings(maxiters+1,n_ini))
  for i in 1:n_ini
    println("$(i) of $(n_ini)")
    losses = []
    Q10s = []
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
    hybrid_opti = DiffEqFlux.sciml_train(loss_L2, p_ini, BFGS(initial_stepnorm = 0.001), maxiters = maxiters, cb = callback)
    Q10_array[1:length(Q10s),i] = Q10s
  end
  Q10_array
end

x_Rb = x_Rb_wiTA
NN = FastChain(FastDense(size(x_Rb,1), 16, tanh),FastDense(16, 16, tanh),FastDense(16, 1,x->x^2))
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