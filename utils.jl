export HyperParameter, Data

import Base
using Flux

mutable struct HyperParameter{T<:AbstractFloat}
    value::Vector{T}
    function HyperParameter(value::Vector{S}) where {S<:AbstractFloat}
        size(value) == (1,) || throw(AssertionError("Size must be 1"))
        new{S}(value)
    end
end
HyperParameter(value::Number) = HyperParameter([float(value)])
Base.convert(::Type{HyperParameter}, value::Number) = HyperParameter([float(value)])
Flux.@functor HyperParameter

for op in (:+, :-, :*, :/, :^)
    @eval Base.$op(x::HyperParameter, y::Number) = Base.$op(promote(x.value[1][1], y)...)
    @eval Base.$op(x::Number, y::HyperParameter) = Base.$op(promote(x, y.value[1][1])...)
end
# Todo make iteration for for hyperparams (i.e. ops like .+)
#for op in (:length)
#    @eval Base.$op(x::HyperParameter) = Base.$op(x.value[1][1])
#end
Base.show(io::IO, x::HyperParameter) = show(io, x.value[1][1])

mutable struct Data
    x::AbstractVecOrMat
    y::AbstractVector
    Data(x, y) = new(x, y)
    Data() = new()
end
Base.convert(::Type{Data}, value::Tuple{AbstractVecOrMat,AbstractVector}) =
    Data(value[1], value[2])
Base.getproperty(d::Data, f::Symbol) = f === :xy ? (d.x, d.y) : getfield(d, f)
function Base.setproperty!(
    d::Data,
    f::Symbol,
    value::Tuple{AbstractVecOrMat,AbstractVector},
)
    if f === :xy
        setfield!(d, :x, value[1])
        setfield!(d, :y, value[2])
    else
        setfield!(d, f, value)
    end
end
Data((x, y)) = Data(x, y)
