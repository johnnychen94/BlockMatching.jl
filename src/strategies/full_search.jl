"""
    FullSearch(d=SqEuclidean(), N; patch_radius, search_radius, search_stride=1)

Construct a block matching algorithm with full search strategy. In some literature, this is also
called _exhaustive search_.

The default patch distance is `Distances.SqEuclidean()`.
"""
struct FullSearch{F,N} <: AbstractBlockMatchingStrategy
    f::F
    patch_radius::CartesianIndex{N}
    search_radius::CartesianIndex{N}
    search_stride::CartesianIndex{N}
end

FullSearch(N; kwargs...) = FullSearch(SqEuclidean(), N; kwargs...)
FullSearch(f, N; patch_radius, search_radius, search_stride=1) =
    FullSearch{typeof(f),N}(f, to_cartesian(N, patch_radius, search_radius, search_stride)...)

function best_match(S::FullSearch, ref, frame, p::CartesianIndex; offset=false)
    size_check(ref, frame)

    rₚ = S.patch_radius
    R_frame = CartesianIndices(frame)
    R_frame = first(R_frame) + rₚ : last(R_frame) - rₚ
    p in R_frame || throw(ArgumentError("Boundary unsupported now: pixel `p = $p` should be within $(first(R_frame)):$(last(R_frame))."))

    R_ref = CartesianIndices(ref)
    R_ref = first(R_ref) + rₚ : last(R_ref) - rₚ
    q_start = max(first(R_ref), p - S.search_radius)
    q_stop = min(last(R_ref), p + S.search_radius)
    candidates = q_start:S.search_stride:q_stop

    patch_p = frame[p - S.patch_radius:p + S.patch_radius]
    dist = map(candidates) do q
        patch_q = @view ref[q - S.patch_radius:q + S.patch_radius]
        S.f(patch_p, patch_q)
    end

    _, idx = findmin(dist)
    q = candidates[idx]

    return offset ? q - p : q
end

function best_match(S::FullSearch, ref, frame; offset=false)
    size_check(ref, frame)

    rₚ = S.patch_radius

    R_frame = CartesianIndices(frame)
    R_ref = CartesianIndices(ref)
    # does not consider boundary
    R_frame = first(R_frame) + rₚ : last(R_frame) - rₚ
    R_ref = first(R_ref) + rₚ : last(R_ref) - rₚ
    matches = OffsetArray(similar(CartesianIndices(R_frame)), rₚ.I)

    # pre-allocation to reduce memeory allocation
    p = CartesianIndex((last(R_frame) + first(R_frame)).I .÷ 2) # use center point to initialize
    q_start = max(first(R_ref), p - S.search_radius)
    q_stop = min(last(R_ref), p + S.search_radius)
    candidates = q_start:S.search_stride:q_stop
    @assert length(candidates) > 0
    patch_p = frame[p-rₚ:p+rₚ] # pre-allocation into a contiguous memory layout
    dist = Vector{Float32}(undef, length(candidates))

    for p in R_frame
        patch_p .= @view frame[p-rₚ:p+rₚ]

        q_start = max(first(R_ref), p - S.search_radius)
        q_stop = min(last(R_ref), p + S.search_radius)
        candidates = q_start:S.search_stride:q_stop
        n = length(candidates)
        for k in 1:n
            q = candidates[k]
            patch_q = @view ref[q-rₚ:q+rₚ]
            dist[k] = S.f(patch_p, patch_q)
        end
        _, idx = findmin(view(dist, 1:n))
        matches[p] = candidates[idx]
    end
    
    offset && (matches .= matches .- OffsetArray(R_frame, rₚ.I))
    return matches
end

function multi_match(S::FullSearch, ref, frame, p::CartesianIndex; num_patches, offset=false)
    size_check(ref, frame)

    rₚ = S.patch_radius
    R_frame = CartesianIndices(frame)
    R_frame = first(R_frame) + rₚ : last(R_frame) - rₚ
    p in R_frame || throw(ArgumentError("Boundary unsupported now: pixel `p = $p` should be within $(first(R_frame)):$(last(R_frame))."))

    R_ref = CartesianIndices(ref)
    R_ref = first(R_ref) + rₚ : last(R_ref) - rₚ
    q_start = max(first(R_ref), p - S.search_radius)
    q_stop = min(last(R_ref), p + S.search_radius)
    candidates = q_start:S.search_stride:q_stop

    patch_p = frame[p - S.patch_radius:p + S.patch_radius]
    T = typeof(S.f(patch_p, patch_p))
    dist = Vector{T}(undef, length(candidates))
    @inbounds for i in 1:length(dist)
        q = candidates[i]
        patch_q = @view ref[q - S.patch_radius:q + S.patch_radius]
        dist[i] = S.f(patch_p, patch_q)
    end

    qs = candidates[partialsortperm(dist, 1:num_patches)]
    return offset ? map(q->q-p, qs) : qs
end

function multi_match(S::FullSearch, ref, frame; num_patches, offset=false)
    size_check(ref, frame)

    rₚ = S.patch_radius

    R_frame = CartesianIndices(frame)
    R_ref = CartesianIndices(ref)
    # does not consider boundary
    R_frame = first(R_frame) + rₚ : last(R_frame) - rₚ
    R_ref = first(R_ref) + rₚ : last(R_ref) - rₚ

    matches = Array{Vector{CartesianIndex{ndims(ref)}}, ndims(ref)}(undef, size(R_frame))
    matches = OffsetArray(matches, rₚ.I)

    # pre-allocation to reduce memeory allocation
    p = CartesianIndex((last(R_frame) + first(R_frame)).I .÷ 2) # use center point to initialize
    q_start = max(first(R_ref), p - S.search_radius)
    q_stop = min(last(R_ref), p + S.search_radius)
    candidates = q_start:S.search_stride:q_stop
    @assert length(candidates) > 0
    patch_p = frame[p-rₚ:p+rₚ] # pre-allocation into a contiguous memory layout
    dist = Vector{Float32}(undef, length(candidates))

    for p in R_frame
        patch_p .= @view frame[p-rₚ:p+rₚ]

        q_start = max(first(R_ref), p - S.search_radius)
        q_stop = min(last(R_ref), p + S.search_radius)
        candidates = q_start:S.search_stride:q_stop
        n = length(candidates)
        for k in 1:n
            q = candidates[k]
            patch_q = @view ref[q-rₚ:q+rₚ]
            dist[k] = S.f(patch_p, patch_q)
        end
        matches[p] = candidates[partialsortperm(view(dist, 1:n), 1:num_patches)]
    end

    if offset
        _offset(p, qs) = map(q->q-p, qs)
        return _offset.(OffsetArray(R_frame, rₚ.I) , matches)
    else
        return matches
    end
end
