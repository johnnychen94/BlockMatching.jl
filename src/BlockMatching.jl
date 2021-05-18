module BlockMatching

using OffsetArrays

export FullSearch, best_match, multi_match

abstract type AbstractBlockMatchingStrategy end

"""
    best_match(S::AbstractBlockMatchingStrategy, ref, [frame=ref], [p::CartesianIndex]; offset=false)

For given pixel `p` in `frame`, find the best match pixel `q` in `ref` with block matching strategy
`S`. If `p` is not provided, it operates on all pixels `p ∈ CartesianIndices(ref)` and returns an array.

!!! note
    For clarification, "pixel" here means the position `CartesianIndex` instead of the value `Colorant`.

# Arguments

* `S::AbstractBlockMatchingStrategy`: required
  The concrete block matching strategy and relevant configs, e.g., patch size, search window size.
  See `subtypes(BlockMatching.AbstractBlockMatchingStrategy)` for possible strategies.
* `ref`::AbstractArray: required
  Reference image where the matched pixel `q` belongs to.
* `frame::AbstractArray`: optional
  The current image/frame where the input pixel `p` belongs to. If `frame` is not provided, it will
  be `ref`.
* `p::CartesianIndex`: optional
  The given pixel `p` that block matching operates on. If not provided, block matching is operating
  on the whole image and thus returns an array of pixels `q`s.

# Parameters

* `offset::Bool`: optional
  if `true`, it returns the motion vector `q-p` instead of the pixel `q`. The default value is
  `false`.
    
# Output

If `p` is given, it returns `CartesianIndex{N}`. Otherwise the block matching operates on
the whole image and returns `Array{CartesianIndex{N}, N}`, i.e., the motion field.

# Examples

If pixel `p` is provided, block matching only operates on the given pixel, the output is the matched
pixels `q`s for `p`.

```jldoctest best_match
using Images, TestImages
using BlockMatching

ref = imresize(testimage("cameraman"), (64, 64))
img = imrotate(ref, 0.2, CartesianIndices(ref).indices)

S = FullSearch(SqEuclidean(), ndims(img), patch_radius=5, search_radius=11)
p = CartesianIndex(17, 17)
best_match(S, img, ref, p)

# output

CartesianIndex(17, 14)
```

Otherwise, the block matching operates on the whole image. Each item of the output array is the
matched pixels `q`s for `p`, i.e., `best_match(S, ref, frame)[p] = best_match(S, ref, frame, p)`.

```jldoctest best_match
julia> matches = best_match(S, img, ref);

julia> summary(matches)
"64×64 $(Matrix{CartesianIndex{2}})"

julia> matches[p] == best_match(S, img, ref, p)
true
```

!!! tip

    `best_match(S, ref, frame)` is usually more performant than `map(p->best_match(S, ref, frame,
    p), CartesianIndices(frame))` because intermediate results can be cached/pre-allocated so as
    to reduce unnecessary computation and memory allocation.

# See also

* [`multi_match`](@ref) can be used when multiple candidates are wanted.
"""
best_match(S::AbstractBlockMatchingStrategy, ref; kwargs...) = best_match(S, ref, ref; kwargs...)
best_match(S::AbstractBlockMatchingStrategy, ref, p::CartesianIndex; kwargs...) = best_match(S, ref, ref, p; kwargs...)

"""
    multi_match(S::AbstractBlockMatchingStrategy, ref, frame=ref, [p::CartesianIndex]; num_patches, offset=true)

For given pixel `p` in `frame`, find matched pixels `q`s in `ref` with block matching strategy `S`.

# Arguments

* `S::AbstractBlockMatchingStrategy`: required
  The concrete block matching strategy and relevant configs, e.g., similarity mesaure, patch size,
  search window size. See `subtypes(BlockMatching.AbstractBlockMatchingStrategy)` for possible
  strategies.
* `ref`::AbstractArray: required
  Reference image where the matched pixel `q` belongs to.
* `frame::AbstractArray`: optional
  The current image/frame where the input pixel `p` belongs to. If `frame` is not provided, it will
  be `ref`.
* `p::CartesianIndex`: optional
  The given pixel `p` that block matching operates on. If not provided, block matching is operating
  on the whole image and thus returns an array of pixels `q`s.

# Parameters

* `num_patches`: required
  The number of matched pixels. If the candidates number is smaller than `num_patches`, the behavior
  is undefined; it may or may not throw errors.
* `offset::Bool`: optional
  if `true`, it returns the motion vector `q-p` instead of the pixel `q`. The default value is
  `false`.

# Output

If `p` is given, it returns `Vector{CartesianIndex{N}}`. Otherwise the block matching operates on
the whole image and returns `Array{Vector{CartesianIndex{N}}, N}`.

# Examples

If pixel `p` is provided, block matching only operates on the given pixel, the output is the matched
pixels `q`s for `p`.

```jldoctest multi_match
using Images, TestImages
using BlockMatching

ref = imresize(testimage("cameraman"), (64, 64))
img = imrotate(ref, 0.2, CartesianIndices(ref).indices)

S = FullSearch(SqEuclidean(), ndims(img), patch_radius=5, search_radius=11)
p = CartesianIndex(17, 17)
multi_match(S, img, ref, p; num_patches=2)

# output

2-element $(Vector{CartesianIndex{2}}):
 CartesianIndex(17, 14)
 CartesianIndex(18, 13)
```

Otherwise, the block matching operates on the whole image. Each item of the output array is the
matched pixels `q`s for `p`, i.e., `multi_match(S, ref, frame)[p] = multi_match(S, ref, frame, p)`.

```jldoctest multi_match
julia> matches = multi_match(S, img, ref; num_patches=2);

julia> summary(matches)
"64×64 $(Matrix{Vector{CartesianIndex{2}}})"

julia> matches[p] == multi_match(S, img, ref, p; num_patches=2)
true
```

!!! tip

    `multi_match(S, ref, frame)` is usually more performant than `map(p->multi_match(S, ref, frame,
    p), CartesianIndices(frame))` because intermediate results can be cached so as to reduce
    computation and memory allocation.

# See also

[`best_match`](@ref) is more performant a choice when `num_patches==1`. It also supports more
sophisticated strategies, e.g., ThreeStepSearch.
"""
multi_match(S::AbstractBlockMatchingStrategy, ref; kwargs...) = multi_match(S, ref, ref; kwargs...)
multi_match(S::AbstractBlockMatchingStrategy, ref, p::CartesianIndex; kwargs...) = multi_match(S, ref, ref, p; kwargs...)


include("utils.jl")
include("strategies/full_search.jl")

end
