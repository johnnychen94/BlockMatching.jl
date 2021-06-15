@testset "FullSearch CUDA" begin
    original = Float32.(imresize(testimage("mri-stack"), (64, 64)));
    ref = original[:, :, 1];
    img = original[:, :, 2];

    @testset "best_match" begin
        cu_ref = CuArray(ref)
        cu_img = CuArray(img)
        S = FullSearch(SqEuclidean(), 2, patch_radius=2, search_radius=5)
        cu_matches = best_match(S, cu_ref, cu_img)
        matches = best_match(S, ref, img)

        # Because the inner loop orders are different, the output results of
        # CPU and CUDA version are different in constant fields.
        @test_broken cu_matches == matches
        @test sum(cu_matches .!= matches)/length(matches) < 0.05

        cu_matches = best_match(S, cu_ref, cu_img; offset=true)
        matches = best_match(S, ref, img; offset=true)
        @test_broken cu_matches == matches
        @test sum(cu_matches .!= matches)/length(matches) < 0.05
    end
end
