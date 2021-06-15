@testset "FullSearch CUDA" begin
    ref = Float32.(imresize(testimage("cameraman"), (64, 64)))
    img = Float32.(imrotate(ref, 0.2, axes(ref)))

    cu_ref = CuArray(ref)
    cu_img = CuArray(img)
    S = FullSearch(SqEuclidean(), 2, patch_radius=2, search_radius=5)
    cu_matches = best_match(S, cu_ref, cu_img)
    matches = best_match(S, ref, img)

    d = mapreduce(+, cu_matches, matches) do cu_q, q
        cityblock(cu_q.I, q.I)
    end
    @test d < 2length(matches)
end
