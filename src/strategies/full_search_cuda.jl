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

    # TODO:
    # we can further share the patch ref[px_start-rₚx:px_end+rₚx, py_start-rₚy:py_end+rₚy]
    # using shared memory so as to reduce global memory access.

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
        ref::CuArray{<:Any,2},
        frame::CuArray{<:Any,2}; offset=false) where F<:DistancesMetrics
    rₚ, Δₛ, rₛ = S.patch_radius, S.search_stride, S.search_radius
    size_check(ref, frame)

    # CuArray{<:RGB} is not supported
    T = Float32
    ref = CUDA.adapt(CuArray{T, 2}, ref)
    frame = CUDA.adapt(CuArray{T, 2}, frame)

    threads = (16, 16)
    blocks = ceil.(Int, size(ref)./threads)

    R_frame = CartesianIndices(size(frame))
    R_frame = first(R_frame)+rₚ:last(R_frame)-rₚ

    cu_matches = OffsetArray(CuArray(fill(CartesianIndex(0, 0), size(R_frame))), rₚ.I)
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks best_match_fullsearch_kernel!(cu_matches, ref, frame, S.f, rₚ, Δₛ, rₛ)
    end

    if offset
        m = parent(cu_matches)
        @. m = m - R_frame
    end
    return cu_matches
end


function multi_match_fullsearch_kernel!(R, ref, frame, f, rₚ, Δₛ, rₛ, K, sizeof_vals)
    T = eltype(frame)
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

    threads = (blockDim().x, blockDim().y)
    vals = @cuDynamicSharedMem T (threads..., K)
    inds = @cuDynamicSharedMem Int16 (threads..., K) offset=sizeof_vals

    idx = (threadIdx().x, threadIdx().y)
    vals = view(vals, idx..., :)
    inds = view(inds, idx..., :)

    # NOTE:
    # For better performance, loop along the row order. This could
    # produce results that different to CPU versions on constant fields.
    @inbounds for px in px_start:Δx:px_end, py in py_start:Δy:py_end
        qx_range = max(rₚx+1, px - rₛx):Δₛx:min(px_end, px + rₛx)
        qy_range = max(rₚy+1, py - rₛy):Δₛy:min(py_end, py + rₛy)
        Rq = CartesianIndices((qx_range, qy_range))

        # Partial sort the smallest K values `vals` and their associated indices `inds`
        # using 1-pass insertion sort with binary search.
        # This is significantly faster than the max-heap sort provided in `src/utils.jl`
        fill!(vals, typemax(eltype(vals)))
        fill!(inds, 0)
        for i = 1:length(Rq)
            qx, qy = Rq[i].I
            x = evaluate_dist(f, ref, frame, px, py, qx, qy, rₚx, rₚy)
            # 1-pass insertion sort
            idx = _insertion_index(x, vals)
            idx == 0 && continue
            # bump the right-hand side and then insert
            for j in K:-1:idx+1
                vals[j] = vals[j-1]
                inds[j] = inds[j-1]
            end
            vals[idx] = x
            inds[idx] = i
        end

        @inbounds for i in 1:K
            R[px, py, i] = Rq[inds[i]]
        end
    end
    return nothing
end

function _insertion_index(x, vals)
    # find the first index such that x < vals[k]
    l, r = 1, length(vals)
    i = (l+r)÷2
    i_old = 1
    @inbounds while i_old != i
        if x < vals[i]
            r = i
        else
            l = i
        end
        i_old = i
        i = (l+r)÷2
    end
    return @inbounds x < vals[l] ? l : x < vals[r] ? r : 0
end

function multi_match(
        S::FullSearch{F},
        ref::CuArray{<:Any, 2},
        frame::CuArray{<:Any, 2}; num_patches, offset=false) where F<:DistancesMetrics

    rₚ, Δₛ, rₛ = S.patch_radius, S.search_stride, S.search_radius
    K = num_patches
    size_check(ref, frame)

    # CuArray{<:RGB} is not supported
    T = Float16
    Tind = Int16
    ref = CUDA.adapt(CuArray{T, 2}, ref)
    frame = CUDA.adapt(CuArray{T, 2}, frame)

    # FIXME: this currently doesn't work with large num_patches
    threads = (8, 4)
    blocks = ceil.(Int, size(frame)./threads)

    function get_shmem(threads, K)
        N = prod(threads)
        sizeof_vals = N * K * sizeof(T)
        sizeof_inds = N * K * sizeof(Tind)
        return sizeof_vals, sizeof_inds
    end

    shmem_sizes = get_shmem(threads, K)
    shmem = sum(shmem_sizes)

    R_frame = CartesianIndices(size(frame))
    R_frame = first(R_frame)+rₚ:last(R_frame)-rₚ
    cu_matches = OffsetArray(CUDA.fill(CartesianIndex(0, 0), size(R_frame)..., K), (rₚ.I..., 0));

    CUDA.@sync begin
        @cuda threads=threads blocks=blocks shmem=shmem multi_match_fullsearch_kernel!(cu_matches, ref, frame, S.f, rₚ, Δₛ, rₛ, K, shmem_sizes[1])
    end

    if offset
        m = parent(cu_matches)
        @. m = m - R_frame
    end
    return cu_matches
end
