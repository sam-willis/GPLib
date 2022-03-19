using LinearAlgebra
using Plots
using Flux
using Flux.Optimise: update!
import Base
import RecipesBase.plot

abstract type Kernel end
abstract type GPModel end

mutable struct HyperParameter{T <: AbstractFloat}
    value:: Vector{T}
    function HyperParameter(value:: Vector{S}) where S <: AbstractFloat
        size(value) == (1,) || throw(AssertionError("Size must be 1"))
        new{S}(value)
    end
end
HyperParameter(value:: Number) = HyperParameter([float(value)])
Base.convert(::Type{HyperParameter}, value :: Number) = HyperParameter([float(value)])
Flux.@functor HyperParameter
# TODO: replace this with a macro or something metaprogrammy
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
Base.show(io::IO, x::HyperParameter) = show(io, x.value[1][1])

struct RBFKernel <: Kernel
    l:: HyperParameter
end
(k:: RBFKernel)(x, y) = exp.(-abs.(x .- y').^2 ./ (2*k.l^2))
Flux.@functor RBFKernel

mutable struct Data
    x:: AbstractVecOrMat
    y:: AbstractVector
    Data(x, y) = new(x, y)
    Data() = new()
end
Base.convert(::Type{Data}, value :: Tuple{AbstractVecOrMat, AbstractVector}) = Data(value[1], value[2])
Base.getproperty(d::Data, f::Symbol) = f === :xy ? (d.x, d.y) : getfield(d, f)
function Base.setproperty!(d::Data, f::Symbol, value :: Tuple{AbstractVecOrMat, AbstractVector})
    if f === :xy 
        setfield!(d, :x, value[1])
        setfield!(d, :y, value[2])
    else
        setfield!(d, f, value)
    end
end
Data((x, y)) = Data(x, y)

struct GPR <: GPModel
    training_data:: Data
    kernel:: Kernel
    σ:: HyperParameter
end
GPR(kernel, σ) = GPR(Data(), kernel, σ)
Flux.trainable(m::GPR) = (m.kernel, m.σ)
Flux.@functor GPR

function predict_mean(model:: GPR, x_new:: AbstractVecOrMat)
    (x, y) = model.training_data.xy
    K = model.kernel
    σ = model.σ
    K(x_new, x) * inv(K(x, x) + I*σ^2) * y
end

function predict_cov(model:: GPR, x_new:: AbstractVecOrMat)
    (x, _) = model.training_data.xy
    K = model.kernel
    σ = model.σ
    K(x_new, x_new) - K(x_new, x) * inv(K(x, x) + I*σ^2) * K(x_new, x)'
end

function predict_std(model:: GPModel, x_new:: AbstractVecOrMat)
    sqrt.(diag(predict_cov(model, x_new)))
end

function loglik(model:: GPR)
    (x, y) = model.training_data.xy
    K = model.kernel
    σ = model.σ
    n = length(x)
    -0.5*y'*inv(K(x, x) + I*σ^2)*y - 0.5*log(det(K(x, x) + I*σ^2)) - 0.5*n*log(2*pi)
end

function grad_step!(model:: GPR, η=1e-3:: AbstractFloat)
    θ = params(model)
    ∇ = gradient(() -> -loglik(model), θ)
    for p in θ
        update!(p, η * ∇[p])
    end
end

# plotting functions

function plot(x:: AbstractVecOrMat, model:: GPR)
    mean = predict_mean(model, x)
    plot(
        x, 
        mean, 
        label="mean"
    )
    std = predict_std(model, x)
    plot!(
        x, 
        mean - std, 
        fillrange=mean + std, 
        alpha=0.2, 
        fillalpha =0.2, 
        c=1, 
        label="var"
    )
    (x_train, y_train) = gpr.training_data.xy
    plot!(
        x_train, 
        y_train, 
        seriestype=:scatter, 
        label = "y"
    )
end

#%% create a problem
x = collect(1:10); 
y = x/10 + 0.5 * rand(size(x, 1))
x_test = collect(1:0.01:20)

gpr = GPR(RBFKernel(1.5), 0.1)
gpr.training_data.xy = (x, y)

#%% basic optimisation
println(params(gpr))
for _=1:10000
    grad_step!(gpr)
end
println(params(gpr))

#%% plot prediction vs training data

plot(x_test, gpr)