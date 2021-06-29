using BlockMatching
using Distances
using ImageCore, ImageTransformations, TestImages, TestImages, ImageDistances
using OffsetArrays
using Test

CUDA_INSTALLED = false
try
    global CUDA_INSTALLED
    # This errors with `IOError` when nvidia driver is not available,
    # in which case we don't even need to try `using CUDA`
    run(pipeline(`nvidia-smi`, stdout=devnull, stderr=devnull))
    push!(LOAD_PATH, "@v#.#") # force using global CUDA installation

    @eval using CUDA
    CUDA.allowscalar(false)
    CUDA_INSTALLED = true
catch e
    e isa Base.IOError || @warn e LOAD_PATH
end
CUDA_FUNCTIONAL = CUDA_INSTALLED && CUDA.functional()
if CUDA_FUNCTIONAL
    @info "CUDA test: enabled"
else
    @warn "CUDA test: disabled"
end

@testset "BlockMatching.jl" begin
    include("utils.jl")
    include("strategies/full_search.jl")

    if CUDA_FUNCTIONAL
        include("strategies/full_search_cuda.jl")
    end
end
