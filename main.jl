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

x_Rb_woTA = @linq dfall |> 
  select(
    #:TA,
    :SW_POT_sm, :SW_POT_sm_diff
  )
x_Rb_woTA = Array(x_Rb_woTA)'
x_Rb_woTA = Float32.(Flux.normalise(x_Rb_woTA))

x_Rb_wiTA = @linq dfall |> 
  select(
    :TA,
    :SW_POT_sm, :SW_POT_sm_diff
  )
x_Rb_wiTA = Array(x_Rb_wiTA)'
x_Rb_wiTA = Float32.(Flux.normalise(x_Rb_wiTA))


NN = FastChain(FastDense(size(x_Rb_woTA,1), 16, tanh),FastDense(16, 16, tanh),FastDense(16, 1,x->x^2))
p_NN_random = Float32.(initial_params(NN))

Rb_NN_random = NN(x_Rb_woTA,p_NN_random)

plot(Rb_NN_random')

#FluxNN = Chain(Dense(size(x_Rb_woTA,1), 16, tanh),Dense(16, 16, tanh),Dense(16, 1,x->x^2))
#Flux.params(FluxNN[1])
#p,NN = Flux.destructure(FluxNN)
#Flux.params(NN[1].W)

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

function test_training(Q10_ini,x_Rb;maxiters = 100,n_ini = 5,withL2 = false)
  Q10_array = Array{Union{Missing,Float32}}(missings(maxiters+1,n_ini)) #Array{Float32}(missingsmaxiters+1,n_ini)
  #loss_array = Array{Union{Missing,Float32}}(missings(maxiters+1,n_ini))

  for i in 1:n_ini
    println("$(i) of $(n_ini)")
    losses = []
    Q10s = []

    callback(θ,l) = begin
      push!(losses, l)
      push!(Q10s,θ[1])
      if length(losses)%5==0
          println("After $(length(losses)) iterations: loss $(losses[end]) | Q10 $(Q10s[end])")
          #println("Current Q10 after $(length(losses)) iterations: $(Q10s[end])")
      end
      false
    end

    NN_for_pars = FastChain(FastDense(size(x_Rb,1), 16, tanh),FastDense(16, 16, tanh),FastDense(16, 1,x->x^2))
    p_NN_random = Float32.(initial_params(NN_for_pars))
    p_ini = [Float32(Q10_ini);p_NN_random]

    function loss_simple(θ,x_Rb;withL2 = withL2)
      mod = fR(Float32.(dfall.TA),x_Rb;p = θ)
      cost = Flux.mse(mod,Float32.(dfall.RECO_syn))
      if withL2 == true
        #println("L2 is on")
        cost = cost + L2(θ[2:end])
      end
      cost
    end

    # make one argument loss
    loss_one_arg(θ) = loss_simple(θ,x_Rb)

    hybrid_opti = DiffEqFlux.sciml_train(loss_one_arg, p_ini, BFGS(initial_stepnorm = 0.1), maxiters = maxiters, cb = callback)
    Q10_array[1:length(Q10s),i] = Q10s
    loss_array[1:length(losses),i] = losses
  end
  (Q10 = Q10_array,loss = loss_array)
end

# without TA in NN
NN = FastChain(FastDense(size(x_Rb_woTA,1), 16, tanh),FastDense(16, 16, tanh),FastDense(16, 1,x->x^2))
Q10_05 = test_training(0.5,x_Rb_woTA)
Q10_15 = test_training(1.5,x_Rb_woTA)
Q10_25 = test_training(2.5,x_Rb_woTA)

p1 = plot(Q10_05.Q10, colour = "blue", labels = "", title = "wo TA in NN")
plot!(Q10_15.Q10, colour = "red", labels = "")
plot!(Q10_25.Q10, colour = "violet", labels = "")

# with TA in NN
x_Rb_wiTA
NN = FastChain(FastDense(size(x_Rb_wiTA,1), 16, tanh),FastDense(16, 16, tanh),FastDense(16, 1,x->x^2))
Q10_05_wiTA = test_training(0.5,x_Rb_wiTA)
Q10_15_wiTA = test_training(1.5,x_Rb_wiTA)
Q10_25_wiTA = test_training(2.5,x_Rb_wiTA)

p2 = plot(Q10_05_wiTA.Q10, colour = "blue", labels = "", title = "wi TA in NN")
plot!(Q10_15_wiTA.Q10, colour = "red", labels = "")
plot!(Q10_25_wiTA.Q10, colour = "violet", labels = "")

# with L2
x_Rb_wiTA
NN = FastChain(FastDense(size(x_Rb_wiTA,1), 16, sigmoid),FastDense(16, 16, sigmoid),FastDense(16, 1,x->x^2))
Q10_05_wiTA_L2 = test_training(0.5,x_Rb_wiTA; withL2 = true)
Q10_15_wiTA_L2 = test_training(1.5,x_Rb_wiTA; withL2 = true)
Q10_25_wiTA_L2 = test_training(2.5,x_Rb_wiTA; withL2 = true)

p3 = plot(Q10_05_wiTA_L2.Q10, colour = "blue", labels = "", title = "wi TA in NN, L2", xlab = "# iterations")
plot(Q10_15_wiTA_L2.Q10, colour = "red", labels = "")
plot!(Q10_25_wiTA_L2.Q10, colour = "violet", labels = "")

gr()
#Plots.scalefontsizes(1.5)
using Plots.PlotMeasures: mm
plot(p1,p2,p3, link=:both, width = 1, layout = (1,3), ylab = "Q10",xlab = "iterations", left_margin=5mm, bottom_margin=5mm)





Reco_opti = fR(dfall.TA,x_Rb;p = hybrid_opti.minimizer)

plot(dfall.DateTime,dfall.RECO_syn)
plot!(dfall.DateTime,Reco_opti)

plot(Q10s)
plot(losses)