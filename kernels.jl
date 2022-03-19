export Kernel, RBFKernel

using Flux

abstract type Kernel end

struct RBFKernel <: Kernel
    l::HyperParameter
end
(k::RBFKernel)(x, y) = exp.(-abs.(x .- y') .^ 2 ./ (2 * k.l^2))
Flux.@functor RBFKernel
