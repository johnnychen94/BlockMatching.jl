using BlockMatching
using Documenter

makedocs(;
    modules=[BlockMatching],
    authors="Johnny Chen <johnnychen94@hotmail.com>",
    repo="https://github.com/johnnychen94/BlockMatching.jl/blob/{commit}{path}#L{line}",
    sitename="BlockMatching.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://johnnychen94.github.io/BlockMatching.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/johnnychen94/BlockMatching.jl",
)
