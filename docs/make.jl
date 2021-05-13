using BlockMatching
using Documenter, DemoCards

format = Documenter.HTML(;
    prettyurls=get(ENV, "CI", "false") == "true",
    canonical="https://johnnychen94.github.io/BlockMatching.jl",
    assets=String[],
)

makedocs(;
    modules=[BlockMatching],
    sitename="BlockMatching.jl",
    format=format,
    pages=[
        "Home" => "index.md",
        "References" => "reference.md"
    ],
)

deploydocs(;
    repo="github.com/johnnychen94/BlockMatching.jl",
)
