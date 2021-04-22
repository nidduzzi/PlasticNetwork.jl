using PlasticNetwork
using Documenter

DocMeta.setdocmeta!(PlasticNetwork, :DocTestSetup, :(using PlasticNetwork); recursive=true)

makedocs(;
    modules=[PlasticNetwork],
    authors="Ahmad Izzuddin",
    repo="https://github.com/nidduzzi/PlasticNetwork.jl/blob/{commit}{path}#{line}",
    sitename="PlasticNetwork.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://nidduzzi.github.io/PlasticNetwork.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/nidduzzi/PlasticNetwork.jl",
)
