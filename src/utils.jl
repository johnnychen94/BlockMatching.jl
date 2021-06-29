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


###
# Min/Max heap-based Top-K algorithm
#
# Modified from https://github.com/JuliaCollections/DataStructures.jl/blob/0230564f0c88a3f65815cb98a8fe78c2658d8db8/src/heaps/arrays_as_heaps.jl
#
# No longer used but kept here in case we still need it in the future.
###
heapleft(i::Integer) = 2i
heapright(i::Integer) = 2i + 1
heapparent(i::Integer) = div(i, 2)

function percolate_down!(vals, inds, i, o::Base.Ordering)
    x, p, len = vals[i], inds[i], length(vals)
    @inbounds while (l = heapleft(i)) <= len
        r = heapright(i)
        j = r > len || Base.Order.lt(o, vals[l], vals[r]) ? l : r
        if Base.Order.lt(o, vals[j], x)
            vals[i] = vals[j]
            inds[i] = inds[j]
            i = j
        else
            break
        end
    end
    vals[i] = x
    inds[i] = p
    return vals, inds
end

function heapify!(vals, inds, o::Base.Ordering)
    for i in heapparent(length(vals)):-1:1
        percolate_down!(vals, inds, i, o)
    end
    return vals, inds
end

function topk!(vals, inds, data, o::Base.Ordering)
    @assert length(vals) == length(inds) <= length(data)
    @inbounds for p = 1:length(data)
        x = data[p]
        if Base.Order.lt(o, first(vals), x)
            vals[1] = x
            inds[1] = p
            heapify!(vals, inds, o)
        end
    end
    return vals, inds
end

"""
    topk(data, K, o=Forward) -> (vals, inds)

Given input `data`, find the largest `K` numbers `vals` and their associated indices `inds`.
If `o=Base.Order.Reverse`, then it's the lowest `K` numbers.

```jldoctest; setup=:(using BlockMatching, Random; Random.seed!(1234))
julia> data = rand(1:20, 20);

julia> vals, inds = BlockMatching.topk(data, 5)
([3, 13, 10, 19, 13], [5, 4, 1, 3, 2])

julia> data[inds] == vals
true
```
"""
function topk(data::AbstractArray{T}, K::Integer, o::Base.Ordering=Base.Forward) where T
    mn, mx = typemin(T), typemax(T)
    fillvalue = Base.Order.lt(o, mn, mx) ? mn : mx
    vals = fill(fillvalue, K)
    inds = fill(0, K)
    return topk!(vals, inds, data, o)
end
