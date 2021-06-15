using .CUDA

function fullsearch_kernel!(R, ref, frame, rₚ, Δₛ, rₛ)
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

    T = eltype(frame)
    # NOTE:
    # For better performance, loop along the row order. This could
    # produce results that different to CPU versions on constant fields.
    @inbounds for px in px_start:Δx:px_end, py in py_start:Δy:py_end
        qx_start = max(rₚx+1, px - rₛx)
        qy_start = max(rₚy+1, py - rₛy)
        qx_end = min(px_end, px + rₛx)
        qy_end = min(py_end, py + rₛy)

        min_val, min_pos = T(Inf), CartesianIndex(qx_start, qy_start)
        for qx in qx_start:Δₛx:qx_end, qy in qy_start:Δₛy:qy_end
            val = zero(T)
            for ox in -rₚx:rₚx, oy in -rₚy:rₚy
                val += abs2(frame[px+ox, py+oy] - ref[qx+ox, qy+oy])
            end
            if val < min_val
                min_val = val
                min_pos = CartesianIndex(qx, qy)
            end
        end
        R[px, py] = min_pos
    end
    return nothing
end

function best_match(
        S::FullSearch{F},
        ref::CuArray{<:Real,2},
        frame::CuArray{<:Real,2}; offset=false) where F<:SqEuclidean
    rₚ, Δₛ, rₛ = S.patch_radius, S.search_stride, S.search_radius
    size_check(ref, frame)

    threads = (16, 16)
    blocks = ceil.(Int, size(ref).÷threads)

    cu_matches = CuArray(fill(CartesianIndex(0, 0), size(frame)))
    @cuda threads=threads blocks=blocks fullsearch_kernel!(cu_matches, ref, frame, rₚ, Δₛ, rₛ)

    R_frame = CartesianIndices(cu_matches)
    R_frame = first(R_frame)+rₚ:last(R_frame)-rₚ
    matches = OffsetArray(Array(view(cu_matches, R_frame)), rₚ.I)

    offset && (matches .= matches .- OffsetArray(R_frame, rₚ.I))
    return matches
end
