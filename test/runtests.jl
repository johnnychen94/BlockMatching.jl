using BlockMatching
using ImageTransformations, TestImages, TestImages, ImageDistances
using Test

use_cuda = false
try
    global use_cuda
    # This errors with `IOError` when nvidia driver is not available,
    # in which case we don't even need to try `using CUDA`
    run(pipeline(`nvidia-smi`, stdout=devnull, stderr=devnull))
    push!(LOAD_PATH, "@v#.#") # force using global CUDA installation

    @eval using CUDA
    use_cuda = true
catch e
    e isa IOError || @warn e LOAD_PATH
end
if use_cuda
    @info "run CUDA test"
else
    @warn "skip CUDA test"
end

@testset "BlockMatching.jl" begin
    include("utils.jl")
    include("strategies/full_search.jl")

    if use_cuda
        include("strategies/full_search_cuda.jl")
    end
end
