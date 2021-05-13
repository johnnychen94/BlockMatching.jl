using BlockMatching
using Images, TestImages
using Test

@testset "BlockMatching.jl" begin
    include("utils.jl")
    include("strategies/full_search.jl")
end
