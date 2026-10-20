# Precompilation workload, run only while the package image is being generated. It needs
# no PrecompileTools dependency. The formula front end always works on a NamedTuple of
# the selected columns, so a NamedTuple workload also covers DataFrame/CSV inputs.

function _precompile_workload()
    n = 24
    s = repeat(1:8, inner = 3)
    y = repeat([1, 0, 0], 8)
    x1 = [sin(1.3i) for i in 1:n]
    x2 = [cos(0.7i) + 0.1i for i in 1:n]
    g = repeat(["a", "b", "c", "a"], 6)
    tbl = (s = s, y = y, x1 = x1, x2 = x2, g = g)
    for f in (@formula(y ~ x1 + x2), @formula(y ~ x1 + g))
        m = fit(ConditionalLogitModel, f, tbl; strata = :s)
        show(devnull, MIME"text/plain"(), m)
        show(devnull, m)
        coeftable(m); confint(m); oddsratios(m); globaltests(m); fitstatistics(m)
        fitted(m)
    end
    m = fit(ConditionalLogitModel, hcat(x1, x2), y, s)
    show(devnull, MIME"text/plain"(), m)
    # m_k = 2 (GLR recursion) path
    y2 = repeat([1, 1, 0, 0], 6); s2 = repeat(1:6, inner = 4)
    fit(ConditionalLogitModel, hcat(x1, x2), y2, s2)
    return nothing
end

if ccall(:jl_generating_output, Cint, ()) == 1
    try
        _precompile_workload()
    catch err
        @debug "ConditionalLogit precompile workload failed" err
    end
end
