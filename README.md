# BlockMatching

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://johnnychen94.github.io/BlockMatching.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://johnnychen94.github.io/BlockMatching.jl/dev)
[![Build Status](https://github.com/johnnychen94/BlockMatching.jl/workflows/CI/badge.svg)](https://github.com/johnnychen94/BlockMatching.jl/actions)
[![Coverage](https://codecov.io/gh/johnnychen94/BlockMatching.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/johnnychen94/BlockMatching.jl)

`BlockMatching` aims to provide a sophisticated implementation on common [block matching
algorithms](https://en.wikipedia.org/wiki/Block-matching_algorithm) for image processing and
computer vision tasks. Block matching is a data and computational intense algorithm, performance is
of high priority for this package.

🚧 This is still a WIP project.

Two functions are provided as the standard API:

- `best_match`: finds the best matching candidate. This is also known as nearest neighbor search.
- `multi_match`: sort the similarities of all candidates and return the smallest K results. This is sometimes known as K nearest neighbor search or top-k selection.

Available block matching strategies:

- `FullSearch`(brute force): search among all possible candidates. This gives the most accurate result 
  but is computationally intensive. CUDA is supported for commonly used distances defined in 
  [Distances.jl].


[Distances.jl]: https://github.com/JuliaStats/Distances.jl
