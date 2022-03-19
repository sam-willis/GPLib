export GPModel, GPR, predict_mean, predict_cov, predict_std, loglik, grad_step!, plot

using LinearAlgebra
using Flux
using Flux.Optimise: update!
using Plots
import RecipesBase.plot

abstract type GPModel end

struct GPR <: GPModel
    training_data::Data
    kernel::Kernel
    σ::HyperParameter
end
GPR(kernel, σ) = GPR(Data(), kernel, σ)
Flux.trainable(m::GPR) = (m.kernel, m.σ)
Flux.@functor GPR

function predict_mean(model::GPR, x_new::AbstractVecOrMat)
    (x, y) = model.training_data.xy
    K = model.kernel
    σ = model.σ
    K(x_new, x) * inv(K(x, x) + I * σ^2) * y
end

function predict_cov(model::GPR, x_new::AbstractVecOrMat)
    (x, _) = model.training_data.xy
    K = model.kernel
    σ = model.σ
    K(x_new, x_new) - K(x_new, x) * inv(K(x, x) + I * σ^2) * K(x_new, x)'
end

function predict_std(model::GPModel, x_new::AbstractVecOrMat)
    sqrt.(diag(predict_cov(model, x_new)))
end

function loglik(model::GPR)
    (x, y) = model.training_data.xy
    K = model.kernel
    σ = model.σ
    n = length(x)
    -0.5 * y' * inv(K(x, x) + I * σ^2) * y - 0.5 * log(det(K(x, x) + I * σ^2)) -
    0.5 * n * log(2 * pi)
end

function grad_step!(model::GPR, η = 1e-3::AbstractFloat)
    θ = params(model)
    ∇ = gradient(() -> -loglik(model), θ)
    for p in θ
        update!(p, η * ∇[p])
    end
end

function plot(x::AbstractVecOrMat, model::GPR)
    mean = predict_mean(model, x)
    plot(x, mean, label = "mean")
    std = predict_std(model, x)
    plot!(
        x,
        mean - std,
        fillrange = mean + std,
        alpha = 0.2,
        fillalpha = 0.2,
        c = 1,
        label = "var",
    )
    (x_train, y_train) = model.training_data.xy
    plot!(x_train, y_train, seriestype = :scatter, label = "y")
end
