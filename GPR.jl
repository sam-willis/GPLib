using Flux
using LinearAlgebra
using Plots

#%% Data structures
mutable struct GPRHyperParameters
    σ:: Number
    l:: Number
end

struct GPR
    K:: Function
    mean:: Function
    cov:: Function
    loglik:: Function
    θ:: GPRHyperParameters
end

#%% Helper functions
RBFKernel(x, y, θ) = exp.(-abs.(x .- y').^2 ./ (2*θ.l^2))

function createGPR(K, θ)
    # extend Kernel.θ with σ and make that GPR.θ
    Kxx_noise(x, θ) = K(x, x, θ) + I*θ.σ^2
    mean(x, y, θ, x_new) = K(x_new, x, θ) * inv(Kxx_noise(x, θ)) * y
    cov(x, θ, x_new) = K(x_new, x_new, θ) - K(x_new, x, θ) * inv(Kxx_noise(x, θ)) * K(x_new, x, θ)'
    loglik(x, y, θ) = -0.5*y'*inv(Kxx_noise(x, θ))*y - 0.5*log(det(Kxx_noise(x, θ))) - 0.5*length(x)*log(2*pi)
    GPR(K, mean, cov, loglik, θ)
end

function grad_step!(g, θ, alpha)
    grads = g(θ)
    for name in fieldnames(typeof(θ))
        grad = getproperty(grads, name)
        value = getproperty(θ, name)
        setproperty!(θ, name, value + alpha * grad)
    end
end

function optimise!(gpr:: GPR, x, y, n_steps=1000, alpha=0.001)
    g(θ) = gradient(θ -> gpr.loglik(x, y, θ), θ)[1]
    for _=1:n_steps
        grad_step!(g, gpr.θ, alpha)
    end
end

#%% create a problem
x = collect(1:10); 
y = x/10 + 0.5 * rand(size(x, 1))
x_test = collect(1:0.01:20)

θ = GPRHyperParameters(0.1, 1) # initial parameters - we gotta start somewhere
gpr = createGPR(RBFKernel, θ)
println(θ)

#%% basic optimisation

optimise!(gpr, x, y)
println(θ)

#%% plot prediction vs training data
mean_test = gpr.mean(x, y, θ, x_test)
std_test = sqrt.(diag(gpr.cov(x, θ, x_test)))

plot(x, y, seriestype = :scatter, label = "y")
plot!(x_test, mean_test, label = "mean")
plot!(x_test, mean_test - std_test, fillrange = mean_test + std_test, alpha=0.2, fillalpha = 0.2, c = 1, label = "var")
