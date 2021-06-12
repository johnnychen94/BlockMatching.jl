using BlockMatching
using ImageTransformations, TestImages, TestImages, ImageDistances
using Test

@testset "BlockMatching.jl" begin
    include("utils.jl")
    include("strategies/full_search.jl")
end
