@testset "FullSearch" begin
    ref = imresize(testimage("cameraman"), (64, 64))
    img = imrotate(ref, 0.2, CartesianIndices(ref).indices)

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
