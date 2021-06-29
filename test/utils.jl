using BlockMatching: to_cartesian
using BlockMatching: topk, topk!

@testset "to_cartesian" begin
    @test @inferred to_cartesian(2, 1, 2) == (CartesianIndex(1, 1), CartesianIndex(2, 2))
    @test @inferred to_cartesian(2, (1, 2), (3, 4)) == (CartesianIndex(1, 2), CartesianIndex(3, 4))
    @test @inferred to_cartesian(2, CartesianIndex(1, 2), CartesianIndex(3, 4)) == (CartesianIndex(1, 2), CartesianIndex(3, 4))
    @test @inferred to_cartesian(2, CartesianIndex(1, 3), 1, (4, 4)) == (CartesianIndex(1, 3), CartesianIndex(1, 1), CartesianIndex(4, 4))

    @test_throws MethodError to_cartesian(3, (1, 2), 3)
    @test_throws MethodError to_cartesian(3, CartesianIndex(1, 2), 3)
end

@testset "topk" begin
    K, N = 20, 200
    for T in (Int, Float32)
        data = T.(rand(1:N, N))
        data_sorted = sort(data)

        vals, inds = topk(data, K)
        @test isempty(setdiff(data_sorted[end-K+1:end], vals))
        @test data[inds] == vals

        inds = fill(0, K)
        vals = fill(typemin(eltype(data)), K)
        topk!(vals, inds, data, Base.Order.Forward)
        @test isempty(setdiff(data_sorted[end-K+1:end], vals))
        @test data[inds] == vals

        vals, inds = topk(data, K, Base.Order.Reverse)
        @test isempty(setdiff(data_sorted[1:K], vals))
        @test data[inds] == vals

        inds = fill(0, K)
        vals = fill(typemax(eltype(data)), K)
        topk!(vals, inds, data, Base.Order.Reverse)
        @test isempty(setdiff(data_sorted[1:K], vals))
        @test data[inds] == vals
    end
end
