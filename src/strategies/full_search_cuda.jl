using .CUDA

@inline function evaluate_dist(f, ref, frame, px, py, qx, qy, rₚx, rₚy)
    val = Distances.eval_start(f, ref, frame)
    @inbounds for ox in -rₚx:rₚx, oy in -rₚy:rₚy
        val = Distances.eval_reduce(f, val, Distances.eval_op(f, frame[px+ox, py+oy], ref[qx+ox, qy+oy]))
    end
    return Distances.eval_end(f, val)
end

function best_match_fullsearch_kernel!(R, ref, frame, f, rₚ, Δₛ, rₛ)
    M, N = size(frame)
    rₚx, rₚy = rₚ.I
    Δₛx, Δₛy = Δₛ.I
    rₛx, rₛy = rₛ.I

    px_start = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    py_start = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    Δx = gridDim().x * blockDim().x
    Δy = gridDim().y * blockDim().y
    px_end = M-rₚx
    py_end = N-rₚy

    if px_start <= rₚx || py_start <= rₚy
        return nothing
    end

    # NOTE:
    # For better performance, loop along the row order. This could
    # produce results that different to CPU versions on constant fields.
    @inbounds for px in px_start:Δx:px_end, py in py_start:Δy:py_end
        qx_range = max(rₚx+1, px - rₛx):Δₛx:min(px_end, px + rₛx)
        qy_range = max(rₚy+1, py - rₛy):Δₛy:min(py_end, py + rₛy)
        min_val, min_pos_x, min_pos_y = eltype(frame)(Inf), first(qx_range), first(qy_range)
        @inbounds for qx in qx_range, qy in qy_range
            val = evaluate_dist(f, ref, frame, px, py, qx, qy, rₚx, rₚy)
            if val < min_val
                min_val = val
                min_pos_x = qx
                min_pos_y = qy
            end
        end
        R[px, py] = CartesianIndex(min_pos_x, min_pos_y)
    end
    return nothing
end

# modified from https://github.com/JuliaStats/Distances.jl/blob/988c92b8b2b6d8a28e1b8aea336f572025ada2f2/src/metrics.jl#L198
# some metrics are removed because they are not supported yet: RogersTanimoto
DistancesMetrics = Union{Euclidean,SqEuclidean,PeriodicEuclidean,Chebyshev,Cityblock,TotalVariation,Minkowski,Hamming,Jaccard,CosineDist,ChiSqDist,KLDivergence,RenyiDivergence,BrayCurtis,JSDivergence,SpanNormDist,GenKLDivergence}

function best_match(
        S::FullSearch{F},
        ref::CuArray{<:Real,2},
        frame::CuArray{<:Real,2}; offset=false) where F<:DistancesMetrics
    rₚ, Δₛ, rₛ = S.patch_radius, S.search_stride, S.search_radius
    size_check(ref, frame)

    threads = (16, 16)
    blocks = ceil.(Int, size(ref)./threads)

    cu_matches = CuArray(fill(CartesianIndex(0, 0), size(frame)))
    @cuda threads=threads blocks=blocks best_match_fullsearch_kernel!(cu_matches, ref, frame, S.f, rₚ, Δₛ, rₛ)

    R_frame = CartesianIndices(cu_matches)
    R_frame = first(R_frame)+rₚ:last(R_frame)-rₚ
    matches = OffsetArray(Array(view(cu_matches, R_frame)), rₚ.I)

    offset && (matches .= matches .- OffsetArray(R_frame, rₚ.I))
    return matches
end
