using Flux
using LinearAlgebra
using Plots
using TypedDelegation
import Base
import Zygote

abstract type Kernel end
abstract type HyperParameters end
abstract type GPModel end

mutable struct HyperParameter
    value # how do I put a type here?
end
Base.:+(x::HyperParameter, y::Number) = Base.:+(promote(x.value[1][1], y)...)
Base.:+(x::Number, y::HyperParameter) = Base.:+(promote(x, y.value[1][1])...)
Base.:-(x::HyperParameter, y::Number) = Base.:-(promote(x.value[1][1], y)...)
Base.:-(x::Number, y::HyperParameter) = Base.:-(promote(x, y.value[1][1])...)
Base.:*(x::HyperParameter, y::Number) = Base.:*(promote(x.value[1][1], y)...)
Base.:*(x::Number, y::HyperParameter) = Base.:*(promote(x, y.value[1][1])...)
Base.:/(x::HyperParameter, y::Number) = Base.:/(promote(x.value[1][1], y)...)
Base.:/(x::Number, y::HyperParameter) = Base.:/(promote(x, y.value[1][1])...)
Base.:^(x::HyperParameter, y::Number) = Base.:^(promote(x.value[1][1], y)...)
Base.:^(x::Number, y::HyperParameter) = Base.:^(promote(x, y.value[1][1])...)
#Base.convert(::Type{T}, param:: HyperParameter{T}) where T = param.value[1][1]
Base.convert(::Type{HyperParameter}, value) = HyperParameter(params([value]))

struct RBFKernel <: Kernel
    l:: HyperParameter
end

(k:: RBFKernel)(x, y) = exp.(-abs.(x .- y').^2 ./ (2*k.l^2))

struct GPR <: GPModel
    kernel:: Kernel
    σ:: HyperParameter
end

function predict_mean(model:: GPR, x, y, x_new)
    K = model.kernel
    σ = model.σ
    K(x_new, x) * inv(K(x, x) + I*σ^2) * y
end

function predict_cov(model:: GPR, x, x_new)
    K = model.kernel
    σ = model.σ
    K(x_new, x_new) - K(x_new, x) * inv(K(x, x) + I*σ^2) * K(x_new, x)'
end

function predict_std(model:: GPModel, x, x_new)
    sqrt.(diag(predict_cov(model, x, x_new)))
end

function loglik(model:: GPR, x, y)
    K = model.kernel
    σ = model.σ
    n = length(x)
    -0.5*y'*inv(K(x, x) + I*σ^2)*y - 0.5*log(det(K(x, x) + I*σ^2)) - 0.5*n*log(2*pi)
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
    g(θ) = gradient(() -> loglik(gpr, x, y), θ.value)
    for _=1:n_steps
        grad_step!(g, gpr.σ, alpha)
    end
end

#%% create a problem
x = collect(1:10); 
y = x/10 + 0.5 * rand(size(x, 1))
x_test = collect(1:0.01:20)

gpr = GPR(RBFKernel(1), 0.1)

#%% basic optimisation

println(gpr.σ)
# optimise!(gpr, x, y)
println(gpr.σ)

#%% plot prediction vs training data
mean_test = predict_mean(gpr, x, y, x_test)
std_test = predict_std(gpr, x, x_test)

plot(x, y, seriestype = :scatter, label = "y")
plot!(x_test, mean_test, label = "mean")
plot!(x_test, mean_test - std_test, fillrange = mean_test + std_test, alpha=0.2, fillalpha = 0.2, c = 1, label = "var")
