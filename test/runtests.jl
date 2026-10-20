include("helpers.jl")

@testset "ConditionalLogit.jl" begin
    @testset "SAS golden output (LOGIEX11)" include("test_sas.jl")
    @testset "R survival::clogit(exact) fixtures" include("test_reference.jl")
    @testset "Likelihood kernels" include("test_kernels.jl")
    @testset "Invariances" include("test_invariance.jl")
    @testset "Front end and edge cases" include("test_edgecases.jl")
    @testset "Separation and convergence" include("test_convergence.jl")
    @testset "StatsAPI and display" include("test_statsapi.jl")
    @testset "Threads and determinism" include("test_threads.jl")
    @testset "Aqua" include("test_aqua.jl")
end
