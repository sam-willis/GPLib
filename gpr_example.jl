include("./gplib.jl")
using .GPLib

using Flux

#%% create a problem
x = collect(1:10);
y = x / 10 + 0.5 * rand(size(x, 1))
x_test = collect(1:0.01:20)

gpr = GPR(RBFKernel(1.5), 0.1)
gpr.training_data.xy = (x, y)

#%% basic optimisation
println(params(gpr))
for _ = 1:10000
    grad_step!(gpr)
end
println(params(gpr))

#%% plot prediction vs training data

plot(x_test, gpr)
