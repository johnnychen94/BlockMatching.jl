@testset "FullSearch CUDA" begin
    original = Float32.(imresize(testimage("mri-stack"), (64, 64)));
    ref = original[:, :, 1];
    img = original[:, :, 2];

    @testset "best_match" begin
        for T in (Float32, N0f8, Gray{Float32}, Gray{N0f8})
            cu_ref = CuArray(T.(ref))
            cu_img = CuArray(T.(img))
            # Although supported, not all metrics are useful for blockmatching purposes
            for d in [
                Euclidean(),
                SqEuclidean(),
                Cityblock(),
                TotalVariation(),
                Minkowski(1.5f0),
            ]
                S = FullSearch(d, 2, patch_radius=2, search_radius=5)
                cu_matches = OffsetArrays.parent_call(Array, best_match(S, cu_ref, cu_img))
                matches = best_match(S, ref, img)

                # Because the inner loop orders are different, the output results of
                # CPU and CUDA version are different in constant fields.
                @test_broken cu_matches == matches
                @test sum(cu_matches .!= matches)/length(matches) < 0.5

                cu_matches = OffsetArrays.parent_call(Array, best_match(S, cu_ref, cu_img; offset=true))
                matches = best_match(S, ref, img; offset=true)
                @test_broken cu_matches == matches
                @test sum(cu_matches .!= matches)/length(matches) < 0.5
            end
        end
    end

    @testset "multi_match" begin
        for T in (Float32, N0f8, Gray{Float32}, Gray{N0f8})
            cu_ref = CuArray(T.(ref))
            cu_img = CuArray(T.(img))

            # Although supported, not all metrics are useful for blockmatching purposes
            for d in [
                Euclidean(),
                SqEuclidean(),
                Cityblock(),
                TotalVariation(),
                Minkowski(1.5f0),
            ]
                S = FullSearch(d, 2, patch_radius=2, search_radius=5)
                num_patches = 20
                cu_matches = OffsetArrays.parent_call(Array, multi_match(S, cu_ref, cu_img; num_patches))
                matches = multi_match(S, ref, img; num_patches)

                # Because the inner loop orders are different, the output results of
                # CPU and CUDA version are different in constant fields.
                @test_broken cu_matches == matches
                diffcount = mapreduce(+, CartesianIndices(axes(matches)[1:2])) do p
                    cu_v = Set(view(cu_matches, p, :))
                    v = Set(view(matches, p, :))
                    length(cu_v) - length(intersect(cu_v, v))
                end
                @test diffcount/length(matches) < 0.02

                cu_matches = OffsetArrays.parent_call(Array, multi_match(S, cu_ref, cu_img; num_patches, offset=true))
                matches = multi_match(S, ref, img; num_patches, offset=true)
                @test_broken cu_matches == matches
                diffcount = mapreduce(+, CartesianIndices(axes(matches)[1:2])) do p
                    cu_v = Set(view(cu_matches, p, :))
                    v = Set(view(matches, p, :))
                    length(cu_v) - length(intersect(cu_v, v))
                end
                @test diffcount/length(matches) < 0.02
            end
        end
    end
end
