using BlockMatching: to_cartesian

@testset "to_cartesian" begin
    @test @inferred to_cartesian(2, 1, 2) == (CartesianIndex(1, 1), CartesianIndex(2, 2))
    @test @inferred to_cartesian(2, (1, 2), (3, 4)) == (CartesianIndex(1, 2), CartesianIndex(3, 4))
    @test @inferred to_cartesian(2, CartesianIndex(1, 2), CartesianIndex(3, 4)) == (CartesianIndex(1, 2), CartesianIndex(3, 4))
    @test @inferred to_cartesian(2, CartesianIndex(1, 3), 1, (4, 4)) == (CartesianIndex(1, 3), CartesianIndex(1, 1), CartesianIndex(4, 4))

    @test_throws MethodError to_cartesian(3, (1, 2), 3)
    @test_throws MethodError to_cartesian(3, CartesianIndex(1, 2), 3)
end
