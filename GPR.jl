using Flux
using LinearAlgebra
using Plots
using TypedDelegation
import Base
import Zygote
using Flux.Optimise: update!
using Base: AbstractVecOrMat, AbstractVector

abstract type Kernel end
abstract type GPModel end

struct HyperParameter{T <: AbstractFloat}
    value:: Vector{T}
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
Base.convert(::Type{HyperParameter}, value :: Number) = HyperParameter([float(value)])
Flux.@functor HyperParameter
Base.show(io::IO, x::HyperParameter) = show(io, x.value[1][1])

struct RBFKernel <: Kernel
    l:: HyperParameter
end
(k:: RBFKernel)(x, y) = exp.(-abs.(x .- y').^2 ./ (2*k.l^2))
Flux.@functor RBFKernel

struct GPR <: GPModel
    training_data:: Tuple{AbstractVecOrMat, AbstractVector}
    kernel:: Kernel
    σ:: HyperParameter
end
Flux.trainable(m::GPR) = (m.kernel, m.σ)
Flux.@functor GPR

function predict_mean(model:: GPR, x_new)
    (x, y) = model.training_data
    K = model.kernel
    σ = model.σ
    K(x_new, x) * inv(K(x, x) + I*σ^2) * y
end

function predict_cov(model:: GPR, x_new)
    (x, _) = model.training_data
    K = model.kernel
    σ = model.σ
    K(x_new, x_new) - K(x_new, x) * inv(K(x, x) + I*σ^2) * K(x_new, x)'
end

function predict_std(model:: GPModel, x_new)
    sqrt.(diag(predict_cov(model, x_new)))
end

function loglik(model:: GPR)
    (x, y) = model.training_data
    K = model.kernel
    σ = model.σ
    n = length(x)
    -0.5*y'*inv(K(x, x) + I*σ^2)*y - 0.5*log(det(K(x, x) + I*σ^2)) - 0.5*n*log(2*pi)
end

function grad_step!(model:: GPR, η=1e-3)
    θ = params(model)
    ∇ = gradient(() -> loglik(model), θ)
    for p in θ
        update!(p, η * ∇[p])
    end
end

#%% create a problem
x = collect(1:10); 
y = x/10 + 0.5 * rand(size(x, 1))
x_test = collect(1:0.01:20)

gpr = GPR((x, y), RBFKernel(1), 0.1)

#%% basic optimisation

println(params(gpr))
for _=1:1000
    grad_step!(gpr)
end
println(params(gpr))

#%% plot prediction vs training data
mean_test = predict_mean(gpr, x_test)
std_test = predict_std(gpr, x_test)

plot(
    x, 
    y, 
    seriestype=:scatter, 
    label = "y"
)
plot!(
    x_test, 
    mean_test, 
    label="mean"
)
plot!(
    x_test, 
    mean_test - std_test, 
    fillrange=mean_test + std_test, 
    alpha=0.2, 
    fillalpha = 0.2, 
    c = 1, 
    label = "var"
)
