@testset "FullSearch" begin
    ref = imresize(testimage("cameraman"), (64, 64))
    img = imrotate(ref, 0.2, CartesianIndices(ref).indices)
    X = repeat(collect(1:9), inner=(3, 7))

    S = FullSearch(SqEuclidean(), 2, patch_radius=1, search_radius=1)
    @test S ==
        FullSearch(2, patch_radius=1, search_radius=1) ==
        FullSearch(2, patch_radius=(1, 1), search_radius=(1, 1)) == 
        FullSearch(2, patch_radius=CartesianIndex(1, 1), search_radius=CartesianIndex(1, 1))

    @testset "correctness test" begin
        S = FullSearch(SqEuclidean(), ndims(X), patch_radius=1, search_radius=1)
        @test_throws ArgumentError best_match(S, X, X, CartesianIndex(1, 1))

        p = best_match(S, X, X, CartesianIndex(3, 3))
        @test p in [CartesianIndex(3, 2), CartesianIndex(3, 3), CartesianIndex(3, 4)]

        p = best_match(S, X, X, CartesianIndex(3, 4))
        @test p in [CartesianIndex(3, 3), CartesianIndex(3, 4), CartesianIndex(3, 5)]
        @test p != CartesianIndex(3, 2) # outside search region

        S = FullSearch(SqEuclidean(), ndims(X), patch_radius=1, search_radius=2)
        p = best_match(S, X, X, CartesianIndex(3, 4))
        @test p in [CartesianIndex(3, 2), CartesianIndex(3, 3), CartesianIndex(3, 4), CartesianIndex(3, 5), CartesianIndex(3, 6)]
        @test p == CartesianIndex(3, 2) # might not always hold, though.

        S = FullSearch(SqEuclidean(), ndims(X), patch_radius=1, search_radius=1)
        p = best_match(S, X, X, CartesianIndex(3, 3), offset=true)
        @test p in [CartesianIndex(0, -1), CartesianIndex(0, 0), CartesianIndex(0, 1)]
        p = best_match(S, X, X, CartesianIndex(3, 4), offset=true)
        @test p in [CartesianIndex(0, -1), CartesianIndex(0, 0), CartesianIndex(0, 1)]
        @test p != CartesianIndex(0, -2)
    end

    @testset "best_match" begin
        S = FullSearch(SqEuclidean(), ndims(img), patch_radius=5, search_radius=11)

        motion_field = best_match(S, img, ref)
        # check if `motion_field[p] == best_match(S, img, ref, p)`
        @test motion_field == map(CartesianIndices(motion_field)) do p
            best_match(S, img, ref, p)
        end
    end

    @testset "multi_match" begin
        S = FullSearch(SqEuclidean(), ndims(img), patch_radius=5, search_radius=11)

        multi_motion_field = multi_match(S, img, ref; num_patches=5)
        # check if `multi_motion_field[p] == multi_match(S, img, ref, p)`
        @test multi_motion_field == map(CartesianIndices(multi_motion_field)) do p
            multi_match(S, img, ref, p; num_patches=5)
        end

        # check if when `num_patches=1`, it has the same result of `best_match`;
        # except that it's array of array
        multi_motion_field = multi_match(S, img, ref; num_patches=1)
        @test map(first, multi_motion_field) == best_match(S, img, ref)
    end
end
