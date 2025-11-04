mutable struct deriv_Chkp{T1,T2}
    S::ShallowWaters.ModelSetup{T1,T2}      # model structure
    J::Float64                              # objective function value
    i::Int                                  # timestep iterator
    t::Int64                                # model time
end

function integrate(chkp, scheme)

    # calculate layer thicknesses for initial conditions
    ShallowWaters.thickness!(chkp.S.Diag.VolumeFluxes.h, chkp.S.Prog.η, chkp.S.forcing.H)
    ShallowWaters.Ix!(chkp.S.Diag.VolumeFluxes.h_u, chkp.S.Diag.VolumeFluxes.h)
    ShallowWaters.Iy!(chkp.S.Diag.VolumeFluxes.h_v, chkp.S.Diag.VolumeFluxes.h)
    ShallowWaters.Ixy!(chkp.S.Diag.Vorticity.h_q, chkp.S.Diag.VolumeFluxes.h)

    # calculate PV terms for initial conditions
    urhs = chkp.S.Diag.PrognosticVarsRHS.u .= chkp.S.Prog.u
    vrhs = chkp.S.Diag.PrognosticVarsRHS.v .= chkp.S.Prog.v
    ηrhs = chkp.S.Diag.PrognosticVarsRHS.η .= chkp.S.Prog.η

    ShallowWaters.advection_coriolis!(urhs, vrhs, ηrhs, chkp.S.Diag, chkp.S)
    ShallowWaters.PVadvection!(chkp.S.Diag, chkp.S)

    # propagate initial conditions
    copyto!(chkp.S.Diag.RungeKutta.u0, chkp.S.Prog.u)
    copyto!(chkp.S.Diag.RungeKutta.v0, chkp.S.Prog.v)
    copyto!(chkp.S.Diag.RungeKutta.η0, chkp.S.Prog.η)

    # store initial conditions of sst for relaxation
    copyto!(chkp.S.Diag.SemiLagrange.sst_ref, chkp.S.Prog.sst)

    # run integration loop with checkpointing
    @ad_checkpoint scheme for chkp.i = 1:chkp.S.grid.nt

        t = chkp.t
        i = chkp.i

        # ghost point copy for boundary conditions
        ShallowWaters.ghost_points!(chkp.S.Prog.u, chkp.S.Prog.v, chkp.S.Prog.η, chkp.S)
        copyto!(chkp.S.Diag.RungeKutta.u1, chkp.S.Prog.u)
        copyto!(chkp.S.Diag.RungeKutta.v1, chkp.S.Prog.v)
        copyto!(chkp.S.Diag.RungeKutta.η1, chkp.S.Prog.η)

        if chkp.S.parameters.compensated
            fill!(chkp.S.Diag.Tendencies.du_sum, zero(chkp.S.parameters.Tprog))
            fill!(chkp.S.Diag.Tendencies.dv_sum, zero(chkp.S.parameters.Tprog))
            fill!(chkp.S.Diag.Tendencies.dη_sum, zero(chkp.S.parameters.Tprog))
        end

        for rki = 1:chkp.S.parameters.RKo
            if rki > 1
                ShallowWaters.ghost_points!(
                    chkp.S.Diag.RungeKutta.u1,
                    chkp.S.Diag.RungeKutta.v1,
                    chkp.S.Diag.RungeKutta.η1,
                    chkp.S
                )
            end

            # type conversion for mixed precision
            u1rhs = chkp.S.Diag.PrognosticVarsRHS.u .= chkp.S.Diag.RungeKutta.u1
            v1rhs = chkp.S.Diag.PrognosticVarsRHS.v .= chkp.S.Diag.RungeKutta.v1
            η1rhs = chkp.S.Diag.PrognosticVarsRHS.η .= chkp.S.Diag.RungeKutta.η1

            ShallowWaters.rhs!(u1rhs, v1rhs, η1rhs, chkp.S.Diag, chkp.S, t)          # momentum only
            ShallowWaters.continuity!(u1rhs, v1rhs, η1rhs, chkp.S.Diag, chkp.S, t)   # continuity equation

            if rki < chkp.S.parameters.RKo
                ShallowWaters.caxb!(
                    chkp.S.Diag.RungeKutta.u1,
                    chkp.S.Prog.u,
                    chkp.S.constants.RKbΔt[rki],
                    chkp.S.Diag.Tendencies.du
                )
                ShallowWaters.caxb!(
                    chkp.S.Diag.RungeKutta.v1,
                    chkp.S.Prog.v,
                    chkp.S.constants.RKbΔt[rki],
                    chkp.S.Diag.Tendencies.dv
                )
                ShallowWaters.caxb!(
                    chkp.S.Diag.RungeKutta.η1,
                    chkp.S.Prog.η,
                    chkp.S.constants.RKbΔt[rki],
                    chkp.S.Diag.Tendencies.dη
                )
            end

            if chkp.S.parameters.compensated
                ShallowWaters.axb!(chkp.S.Diag.Tendencies.du_sum, chkp.S.constants.RKaΔt[rki], chkp.S.Diag.Tendencies.du)
                ShallowWaters.axb!(chkp.S.Diag.Tendencies.dv_sum, chkp.S.constants.RKaΔt[rki], chkp.S.Diag.Tendencies.dv)
                ShallowWaters.axb!(chkp.S.Diag.Tendencies.dη_sum, chkp.S.constants.RKaΔt[rki], chkp.S.Diag.Tendencies.dη)
            else
                ShallowWaters.axb!(
                    chkp.S.Diag.RungeKutta.u0,
                    chkp.S.constants.RKaΔt[rki],
                    chkp.S.Diag.Tendencies.du
                )
                ShallowWaters.axb!(
                    chkp.S.Diag.RungeKutta.v0,
                    chkp.S.constants.RKaΔt[rki],
                    chkp.S.Diag.Tendencies.dv
                )
                ShallowWaters.axb!(
                    chkp.S.Diag.RungeKutta.η0,
                    chkp.S.constants.RKaΔt[rki],
                    chkp.S.Diag.Tendencies.dη
                )
            end
        end

        if chkp.S.parameters.compensated
            ShallowWaters.axb!(chkp.S.Diag.Tendencies.du_sum, -1, chkp.S.Diag.Tendencies.du_comp)
            ShallowWaters.axb!(chkp.S.Diag.Tendencies.dv_sum, -1, chkp.S.Diag.Tendencies.dv_comp)
            ShallowWaters.axb!(chkp.S.Diag.Tendencies.dη_sum, -1, chkp.S.Diag.Tendencies.dη_comp)

            ShallowWaters.axb!(chkp.S.Diag.RungeKutta.u0, 1, chkp.S.Diag.Tendencies.du_sum)
            ShallowWaters.axb!(chkp.S.Diag.RungeKutta.v0, 1, chkp.S.Diag.Tendencies.dv_sum)
            ShallowWaters.axb!(chkp.S.Diag.RungeKutta.η0, 1, chkp.S.Diag.Tendencies.dη_sum)

            ShallowWaters.dambmc!(
                chkp.S.Diag.Tendencies.du_comp,
                chkp.S.Diag.RungeKutta.u0,
                chkp.S.Prog.u,
                chkp.S.Diag.Tendencies.du_sum
            )
            ShallowWaters.dambmc!(
                chkp.S.Diag.Tendencies.dv_comp,
                chkp.S.Diag.RungeKutta.v0,
                chkp.S.Prog.v,
                chkp.S.Diag.Tendencies.dv_sum
            )
            ShallowWaters.dambmc!(
                chkp.S.Diag.Tendencies.dη_comp,
                chkp.S.Diag.RungeKutta.η0,
                chkp.S.Prog.η,
                chkp.S.Diag.Tendencies.dη_sum
            )
        end

        ShallowWaters.ghost_points!(
            chkp.S.Diag.RungeKutta.u0,
            chkp.S.Diag.RungeKutta.v0,
            chkp.S.Diag.RungeKutta.η0,
            chkp.S
        )

        u0rhs = chkp.S.Diag.PrognosticVarsRHS.u .= chkp.S.Diag.RungeKutta.u0
        v0rhs = chkp.S.Diag.PrognosticVarsRHS.v .= chkp.S.Diag.RungeKutta.v0
        η0rhs = chkp.S.Diag.PrognosticVarsRHS.η .= chkp.S.Diag.RungeKutta.η0

        if chkp.S.parameters.dynamics == "nonlinear" && chkp.S.grid.nstep_advcor > 0 && (i % chkp.S.grid.nstep_advcor) == 0
            ShallowWaters.UVfluxes!(u0rhs, v0rhs, η0rhs, chkp.S.Diag, chkp.S)
            ShallowWaters.advection_coriolis!(u0rhs, v0rhs, η0rhs, chkp.S.Diag, chkp.S)
        end

        if (chkp.i % chkp.S.grid.nstep_diff) == 0
        ShallowWaters.bottom_drag!(u0rhs, v0rhs, η0rhs, chkp.S.Diag, chkp.S)
        ShallowWaters.diffusion!(u0rhs, v0rhs, chkp.S.Diag, chkp.S)
        ShallowWaters.add_drag_diff_tendencies!(
            chkp.S.Diag.RungeKutta.u0,
            chkp.S.Diag.RungeKutta.v0,
            chkp.S.Diag,
            chkp.S
        )
        ShallowWaters.ghost_points_uv!(
            chkp.S.Diag.RungeKutta.u0,
            chkp.S.Diag.RungeKutta.v0,
            chkp.S
        )
    end

    t += chkp.S.grid.dtint

    u0rhs = chkp.S.Diag.PrognosticVarsRHS.u .= chkp.S.Diag.RungeKutta.u0
    v0rhs = chkp.S.Diag.PrognosticVarsRHS.v .= chkp.S.Diag.RungeKutta.v0
    ShallowWaters.tracer!(i, u0rhs, v0rhs, chkp.S.Prog, chkp.S.Diag, chkp.S)

    copyto!(chkp.S.Prog.u, chkp.S.Diag.RungeKutta.u0)
    copyto!(chkp.S.Prog.v, chkp.S.Diag.RungeKutta.v0)
    copyto!(chkp.S.Prog.η, chkp.S.Diag.RungeKutta.η0)

    end

    temp = ShallowWaters.PrognosticVars{Float64}(ShallowWaters.remove_halo(
        chkp.S.Prog.u,
        chkp.S.Prog.v,
        chkp.S.Prog.η,
        chkp.S.Prog.sst,
        chkp.S
    )...)

    chkp.J = (sum(temp.u.^2) + sum(temp.v.^2))
    return chkp.J

end

function compute_derivative(Ndays)

    # Type precision
    T = Float64

    P = ShallowWaters.Parameter(T=T; output=false,
        L_ratio=1,
        g=9.81,
        H=500,
        wind_forcing_x="double_gyre",
        Lx=3840e3,
        seasonal_wind_x=false,
        topography="flat",
        bc="nonperiodic",
        bottom_drag="quadratic",
        tracer_advection=false,
        tracer_relaxation=false,
        α=2,
        nx=128,
        Ndays=Ndays,
        initial_cond="ncfile",
        initpath="128_10yearspinup_fromrest_noslipbc_epsetup/"
    )
    S = ShallowWaters.model_setup(P)

    snaps = Int(floor(sqrt(S.grid.nt)))
    revolve = Revolve(
        snaps;
        verbose=1,
        gc=true,
        write_checkpoints=false,
        write_checkpoints_filename = "",
        write_checkpoints_period = 2274
    )

    chkp = deriv_Chkp{T, T}(S,
        0.0,
        1,
        0.0
    )
    dchkp = Enzyme.make_zero(chkp)

    J = @time autodiff(
        set_runtime_activity(Enzyme.ReverseWithPrimal),
        energy_integration,
        Active,
        Duplicated(chkp, dchkp),
        Const(revolve)
    )[2]

    return chkp, dchkp

end

function finite_difference()

    # corresponds to x = 600km on the ugrid with halo
    x_coord = 22

    # corresponds to y = 2190km on the vgrid with halo
    y_coord = 75

    # Type precision
    T = Float64

    Ndays = 10

    primal, enzyme_deriv = compute_derivative(Ndays)

    P = ShallowWaters.Parameter(T=T;
        output=false,
        L_ratio=1,
        g=9.81,
        H=500,
        wind_forcing_x="double_gyre",
        Lx=3840e3,
        seasonal_wind_x=false,
        topography="flat",
        bc="nonperiodic",
        bottom_drag="quadratic",
        tracer_advection=false,
        tracer_relaxation=false,
        α=2,
        nx=128,
        Ndays=Ndays,
        initial_cond="ncfile",
        initpath="128_10yearspinup_fromrest_noslipbc_epsetup/"
    )

    S0 = ShallowWaters.model_setup(P)

    S2 = deepcopy(S0)
    chkp2 = deriv_Chkp{T, T}(S2,
        0.0,
        1,
        0.0
    )

    snaps = Int(floor(sqrt(S2.grid.nt)))
    revolve = Revolve(
        snaps;
        verbose=1,
        gc=true,
        write_checkpoints=false,
        write_checkpoints_filename = "",
        write_checkpoints_period = 2274
    )

    @time unperturbed_loss = integrate(chkp2, revolve)

    s = 1
    diffsu = []
    for j = 1:128

        S3 = deepcopy(S0)
        chkp3 = deriv_Chkp{T, T}(S3,
            0.0,
            1,
            0.0
        )

        chkp3.S.Prog.u[x_coord, j+2] += s

        J = integrate(chkp3, revolve)
        push!(diffsu, (J - unperturbed_loss) / s)

    end

    s = 1
    diffsv = []
    for j = 1:128

        S3 = deepcopy(S0)
        chkp3 = deriv_Chkp{T, T}(S3,
            0.0,
            1,
            0.0
        )

        chkp3.S.Prog.v[j+2, y_coord] += s

        J = integrate(chkp3, revolve)
        push!(diffsv, (J - unperturbed_loss) / s)

    end

    # states = ShallowWaters.PrognosticVars{Float64}(ShallowWaters.remove_halo(
    #             chkp.S.Prog.u,
    #             chkp.S.Prog.v,
    #             chkp.S.Prog.η,
    #             chkp.S.Prog.sst,
    #             chkp.S
    # )...)
    dstates = ShallowWaters.PrognosticVars{Float64}(ShallowWaters.remove_halo(
                enzyme_deriv.S.Prog.u,
                enzyme_deriv.S.Prog.v,
                enzyme_deriv.S.Prog.η,
                enzyme_deriv.S.Prog.sst,
                primal.S
    )...)

    # states = ShallowWaters.PrognosticVars{Float64}(ShallowWaters.remove_halo(
    #             chkp.S.Prog.u,
    #             chkp.S.Prog.v,
    #             chkp.S.Prog.η,
    #             chkp.S.Prog.sst,
    #             chkp.S
    # )...)
    # dstates = ShallowWaters.PrognosticVars{Float64}(ShallowWaters.remove_halo(
    #             dchkp.S.Prog.u,
    #             dchkp.S.Prog.v,
    #             dchkp.S.Prog.η,
    #             dchkp.S.Prog.sst,
    #             chkp.S
    # )...)

    # chkp = load_object("results/primal_10day_energyloss_100625.jld2")
    # dchkp = load_object("results/deriv_10day_energyloss_100625.jld2")

    # diffsv = load_object("./results/vderiv_finitediffs_10daycheck_100625.jld2")
    # diffsu = load_object("./results/uderiv_finitediffs_10daycheck_100625.jld2")

    fig = Figure(fontsize=17, size=(1000,450));
    ax1 = Axis(fig[1,1],
        xlabel="y (km)",
        ylabel=L"\partial J / \partial u(t_0, 22, y)"
    )
    ax2 = Axis(fig[2,1],
        xlabel="x (km)",
        ylabel=L"\partial J / \partial v(t_0, x, 75)"
    )
    scale_inv = primal.S.constants.scale_inv
    scale = primal.S.constants.scale

    scatter!(ax1, LinRange(0, 3840, 128), (dstates.u[x_coord-2, :]), label="Enzyme derivative")
    scatter!(ax1, LinRange(0, 3840, 128), (diffsu) ./ (scale * 128^2), label="Finite difference approximation", marker=:cross)
    axislegend(ax1, position=:lt)
    scatter!(ax2, LinRange(0, 3840, 128), (dstates.v[:, y_coord-2]), label="Enzyme derivative")
    scatter!(ax2, LinRange(0, 3840, 128), (scale_inv .* diffsv) ./ (128^2), label="Finite difference approximation",marker=:cross)
    axislegend(ax2)

    ga = fig[1, 1] = GridLayout()
    gb = fig[2, 1] = GridLayout()
    for (label, layout) in zip(["(a)", "(b)"], [ga, gb])
    Label(layout[1, 1, TopLeft()], label,
        fontsize = 15,
        font = :bold,
        padding = (0, 5, 5, 0),
        halign = :right)
    end

    # modified initial condition derivative figure, now includes dashed lines to indicate where
    # finite-differences were done

    fig = Figure(size=(900, 350), fontsize=15);
    ax1, hm1 = heatmap(fig[1,1], LinRange(0, 3840, 128),
    LinRange(0, 3840, 128),
    dstates.u,
    colormap=:balance,
    axis=(xlabel="km", ylabel="km", title=L"\partial J / u(t_0)"),
    colorrange=(-maximum(dstates.u),
    maximum(dstates.u))
    );
    lines!(ax1, 600 .* ones(128), LinRange(0, 3840, 128), color=:purple1, linestyle=:dash)
    Colorbar(fig[1,2], hm1)


    ax2, hm2 = heatmap(fig[1,3], LinRange(0, 3840, 128),
    LinRange(0, 3840, 128),
    dstates.v,
    colormap=:balance,
    axis=(xlabel="km", ylabel="km", title=L"\partial J / v(t_0)"),
    colorrange=(-maximum(dstates.v),
    maximum(dstates.v))
    );
    lines!(ax2, LinRange(0, 3840, 128), 2190 .* ones(128), color=:purple1, linestyle=:dash)
    Colorbar(fig[1,4], hm2)

    ga = fig[1, 1] = GridLayout()
    gb = fig[1, 3] = GridLayout()
    for (label, layout) in zip(["(a)", "(b)"], [ga, gb])
    Label(layout[1, 1, TopLeft()], label,
        fontsize = 15,
        font = :bold,
        padding = (0, 5, 5, 0),
        halign = :right)
    end

end
