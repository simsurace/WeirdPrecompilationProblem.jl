module WeirdPrecompilationProblem

using LinearAlgebra
using Optim
using ParameterHandling
using Zygote

export S, init, grad, minimize

struct S
    a::Vector{Float64}
    b::Vector{Float64}
end

init(s::S) = (
    0.,
    positive(1.), positive(1.),
    positive(1.),
    positive(1.), positive(1.), positive(1.),
    positive(1.), positive(1.), positive(1.),
    bounded(1., 0.5, 1.5),
    fixed(2.),
    rand(length(s.a)),
    positive(rand(length(s.b))),
    positive_definite(Matrix{Float64}(I, length(s.b), length(s.b))),
)

function (s::S)(θ)
    return θ[1]^2 + θ[2] * θ[3] + θ[4] + θ[5] * θ[6] * θ[7] + θ[8] * θ[9] * θ[10] +
        θ[11]^2 + θ[12] + dot(θ[13] .^ 2, s.a) + dot(θ[14] .^ 2, s.b) + dot(s.b, θ[15], s.b)
end

grad(S, θ) = only(gradient(S, θ))
grad(S, par, unflatten) = only(gradient(x -> S(unflatten(x)), par))

function minimize(s::S, θ)
    par, unflatten = value_flatten(θ)
    res = optimize(s ∘ unflatten, _par -> grad(s, _par, unflatten), par, BFGS(); inplace = false)
    return Optim.minimizer(res)
end

using PrecompileTools

@setup_workload begin
    s = S(rand(10), rand(30))
    θ = init(s)
    @info "Begin custom precompilation workload"
    @compile_workload begin
        @time minimize(s, θ)
    end
    @info "Finished custom precompilation workload"
end

end # module WeirdPrecompilationProblem
