"""
    to_cartesian(N, args::Index...)

Convert `args` to a tuple of `CartesianIndex{N}`. Each item in `args` is either `Int`, `NTuple{N,
Int}` or `CartesianIndex{N}`; if it is `Int` it will be repeated, i.e., `ntuple(_->i, N)`.

# Examples

```jldoctest; setup=:(using BlockMatching: to_cartesian)
julia> to_cartesian(2, 3, 1, 4)
(CartesianIndex(3, 3), CartesianIndex(1, 1), CartesianIndex(4, 4))

julia> to_cartesian(2, CartesianIndex(1, 3), 1, (4, 4))
(CartesianIndex(1, 3), CartesianIndex(1, 1), CartesianIndex(4, 4))
```
"""
to_cartesian(N::Int, args...) = to_cartesian(Val(N), args...)
to_cartesian(::Val{N}, args::Union{Int, NTuple{N, Int}, CartesianIndex{N}}...) where N = map(i->_to_cartesian(Val(N), i), args)
_to_cartesian(::Val{N}, i::Int) where N = __to_cartesian(ntuple(_->i, N))
_to_cartesian(::Val{N}, t::NTuple{N, Int}) where N = __to_cartesian(t)
_to_cartesian(::Val{N}, t::CartesianIndex{N}) where N = t
__to_cartesian(t::NTuple{N, Int}) where N = CartesianIndex(t)


function size_check(ref, frame)
    Base.require_one_based_indexing(ref)
    Base.require_one_based_indexing(frame)
    any(size(ref) .< size(frame)) && throw(ArgumentError("`size(ref) = $(size(ref))` should be larger than `size(frame) = $(size(frame))`."))
end
