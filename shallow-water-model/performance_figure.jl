# mutable struct energy_Chkp{T1,T2}
#     S::ShallowWaters.ModelSetup{T1,T2}      # model structure
#     J::Float64                              # objective function value
#     i::Int                                  # timestep iterator
#     t::Int64                                # model time
# end
# energy_integration(chkp, scheme)
# energy_integration_nocp(chkp)
# time to compile derivative code for the energy loss function: 107 seconds
# size per checkpoint: 12.857 MiB

function compute_enzyme_times()

    # Type precision
    T = Float64

    Ndays = [1, 3, 5, 10, 20]
    times = []
    for days in Ndays
    
        P = ShallowWaters.Parameter(T=T,
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
            Ndays=days,
            initial_cond="ncfile",
            initpath="128_10yearspinup_fromrest_noslipbc_epsetup/"
        )

        S = ShallowWaters.model_setup(P)

        model = energy_Chkp{T, T}(S, 0.0, 1, 0.0)
        dmodel = Enzyme.make_zero(model)
        t = @elapsed autodiff(
           set_runtime_activity(Enzyme.ReverseWithPrimal),
           energy_integration_nocp,
           Active,
           Duplicated(model, dmodel)
        )

        push!(times, t)

    end

    return times

end

function compute_cp_times()

    # Type precision
    T = Float64

    Ndays = [1, 3, 5, 10, 20, 30, 35, 40, 50, 70, 100]
    times = []
    for days in Ndays
    
        P = ShallowWaters.Parameter(T=T,
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
            Ndays=days,
            initial_cond="ncfile",
            initpath="128_10yearspinup_fromrest_noslipbc_epsetup/"
        )

        S = ShallowWaters.model_setup(P)

        snaps = Int(floor(sqrt(S.grid.nt)))
        revolve = Revolve(
            snaps;
            verbose=1,
            gc=true
        )

        model = energy_Chkp{T, T}(S, 0.0, 1, 0.0)
        dmodel = Enzyme.make_zero(model)

        t = @elapsed autodiff(
           set_runtime_activity(Enzyme.ReverseWithPrimal),
           energy_integration,
           Active,
           Duplicated(model, dmodel),
           Const(revolve)
        )

        # t = @elapsed ShallowWaters.time_integration(S)

        push!(times, t)

    end

    return times

end

function make_figure()

    # enzyme_times = compute_enzyme_times()
    # cp_enzyme_times = compute_cp_times()

    enzyme_times = load_object("results/enzyme_times_1-3-5-10-20days.jld2")
    cp_enzyme_times = load_object("results/cptimes_enzyme+30-35-40-50-70-100-365days.jld2")

    enzyme_days = [1, 3, 5, 10, 20]
    cp_days = [1, 3, 5, 10, 20, 30, 35, 40, 50, 70, 100]

    timesteps = []
    num_of_chkps = []
    for days in cp_days

        P = ShallowWaters.Parameter(T=T,
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
        Ndays=days,
        initial_cond="ncfile",
        initpath="128_10yearspinup_fromrest_noslipbc_epsetup/"
        )

        S = ShallowWaters.model_setup(P)

        push!(timesteps, S.grid.nt)
        push!(num_of_chkps, Int(floor(sqrt(S.grid.nt))))

    end

    fig = Figure(fontsize=17, size=(1000,450));
    ax1 = Axis(fig[1,1],
        xlabel="Timesteps",
        ylabel="Execution time (s)",
        xscale=log10,
        yscale=log10
    )
    scatterlines!(ax1, timesteps, cp_enzyme_times[1:end-1], label="With checkpointing")
    scatterlines!(ax1, timesteps[1:5], enzyme_times, label="Without checkpointing")
    sideinfo1 = Label(fig[0, 0], "(a)", font=:bold, fontsize=20)
    axislegend(position = :rb)

    ax2 = Axis(fig[1, 3],
        xlabel="Timesteps",
        ylabel="Memory Utilization (MiB)",
        xscale=log10,
        yscale=log10
    )
    scatterlines!(ax2, timesteps, [x*12.857 for x in num_of_chkps], label="With checkpointing")
    scatterlines!(ax2, timesteps[1:5], [x*12.857 for x in timesteps[1:5]], label="Without checkpointing")
    sideinfo2 = Label(fig[0, 2], "(b)",font=:bold,fontsize=20)
    axislegend(position=:rb)

    # cost over time
    fig = Figure(fontsize=17);
    ax = Axis(fig[1,1],
        xlabel="Iterations",
        ylabel="Loss value (m/s)",
        yscale=log10
    )
    lines!(ax, iterations, cost_over_time_uandv_10days)

    # Execution time, execution time rel to forward run
    # memory utilization, memory utilization rel to forward run
    fig = Figure(fontsize=17, size=(850,750));
    ax1 = Axis(fig[1,1],
        xlabel="Iterations",
        ylabel="Time (s)",
        title="Execution time",
        yscale=log10,
        xscale=log10
    )
    scatterlines!(ax1, timesteps, cp_times[1:end-1], label="With checkpointing")
    # scatterlines!(ax1, timesteps, cpfixed_times, label="With checkpointing, fixed checkpoints")
    scatterlines!(ax1, timesteps[1:5], enzyme_times, label="Without checkpointing")
    scatterlines!(ax1, timesteps, forward_times, label="Forward run")
    # Legend(fig[1,2], ax1)

    ax2 = Axis(fig[1,2],
        xlabel="Iterations",
        ylabel="Time (s)",
        title="Execution time relative to forward run",
        xscale=log10
    )
    scatterlines!(ax2, timesteps, cp_times[1:end-1] ./ forward_times[1:11], label="With checkpointing")
    # scatterlines!(ax2, timesteps, cpfixed_times ./ forward_times[1:11], label="With checkpointing, fixed checkpoints")
    scatterlines!(ax2, timesteps[1:5], enzyme_times ./ forward_times[1:5], label="Without checkpointing")
    # Legend(fig[1,4], ax2)

    ax3 = Axis(fig[2,1],
        xlabel="Iterations",
        ylabel="MiB",
        title="Memory utilization",
        xscale=log10,
        yscale=log10
    )
    scatterlines!(ax3, timesteps, [x*12.857 for x in num_of_chkps], label="With checkpointing")
    # scatterlines!(ax3, timesteps, [x*12.857 for x in nt_chkps], label="With checkpointing, fixed checkpoints")
    scatterlines!(ax3, timesteps[1:5], [x*12.857 for x in timesteps[1:5]], label="Without checkpointing")
    scatterlines!(ax3, timesteps, [.5287 for x in num_of_chkps], label="Forward run")
    # Legend(fig[2,2], ax3)

    ax4 = Axis(fig[2,2],
        xlabel="Iterations",
        ylabel="MiB",
        title="Memory utilized relative to forward run",
        xscale=log10
    )
    scatterlines!(ax4, timesteps, [x*12.857 for x in num_of_chkps] ./ .5287, label="With checkpointing")
    # scatterlines!(ax4, timesteps, [x*12.857 for x in nt_chkps] ./ .5287, label="With checkpointing, fixed checkpoints")
    scatterlines!(ax4, timesteps[1:5], [x*12.857 for x in timesteps[1:5]] ./ .5287, label="Without checkpointing")
    # Legend(fig[2,4], ax4)

    Legend(fig[3,1:2], ax1, orientation = :horizontal)

    ga = fig[1, 1] = GridLayout()
    gb = fig[1, 2] = GridLayout()
    gc = fig[2, 1] = GridLayout()
    gd = fig[2, 2] = GridLayout()
    for (label, layout) in zip(["(a)", "(b)", "(c)", "(d)"], [ga, gb, gc, gd])
    Label(layout[1, 1, TopLeft()], label,
        fontsize = 15,
        font = :bold,
        padding = (0, 5, 5, 0),
        halign = :right)
    end


    fig = Figure(fontsize=17, size=(1200,375));
    ax1 = Axis(fig[1,1],
        xlabel="Iterations",
        ylabel="Time (s)",
        title="Execution time",
        yscale=log10,
        xscale=log10
    )
    scatterlines!(ax1, timesteps, cp_times[1:end-1], label="With checkpointing")
    # scatterlines!(ax1, timesteps, cpfixed_times, label="With checkpointing, fixed checkpoints")
    scatterlines!(ax1, timesteps[1:5], enzyme_times, label="Without checkpointing")
    scatterlines!(ax1, timesteps, forward_times, label="Forward run")
    # Legend(fig[1,2], ax1)

    ax2 = Axis(fig[1,2],
        xlabel="Iterations",
        ylabel="Time (s)",
        title="Execution time relative to forward run",
        xscale=log10
    )
    scatterlines!(ax2, timesteps, cp_times[1:end-1] ./ forward_times[1:11], label="With checkpointing")
    # scatterlines!(ax2, timesteps, cpfixed_times ./ forward_times[1:11], label="With checkpointing, fixed checkpoints")
    scatterlines!(ax2, timesteps[1:5], enzyme_times ./ forward_times[1:5], label="Without checkpointing")
    # Legend(fig[1,4], ax2)

    ax3 = Axis(fig[1,3],
        xlabel="Iterations",
        ylabel="MiB",
        title="Memory utilization",
        xscale=log10,
        yscale=log10
    )
    scatterlines!(ax3, timesteps, [x*12.857 for x in num_of_chkps], label="With checkpointing")
    # scatterlines!(ax3, timesteps, [x*12.857 for x in nt_chkps], label="With checkpointing, fixed checkpoints")
    scatterlines!(ax3, timesteps[1:5], [x*12.857 for x in timesteps[1:5]], label="Without checkpointing")
    scatterlines!(ax3, timesteps, [.5287 for x in num_of_chkps], label="Forward run")
    # Legend(fig[2,2], ax3)

    ax4 = Axis(fig[1,4],
        xlabel="Iterations",
        ylabel="MiB",
        title="Memory utilized relative to forward run",
        xscale=log10
    )
    scatterlines!(ax4, timesteps, [x*12.857 for x in num_of_chkps] ./ .5287, label="With checkpointing")
    # scatterlines!(ax4, timesteps, [x*12.857 for x in nt_chkps] ./ .5287, label="With checkpointing, fixed checkpoints")
    scatterlines!(ax4, timesteps[1:5], [x*12.857 for x in timesteps[1:5]] ./ .5287, label="Without checkpointing")
    # Legend(fig[2,4], ax4)

    Legend(fig[2,2:3], ax1, orientation = :horizontal)

    ga = fig[1, 1] = GridLayout()
    gb = fig[1, 2] = GridLayout()
    gc = fig[1, 3] = GridLayout()
    gd = fig[1, 4] = GridLayout()
    for (label, layout) in zip(["(a)", "(b)", "(c)", "(d)"], [ga, gb, gc, gd])
    Label(layout[1, 1, TopLeft()], label,
        fontsize = 15,
        font = :bold,
        padding = (0, 5, 5, 0),
        halign = :right)
    end
end

function cost_and_iters()
    cost_over_time_uandv_5days = [1.0796540e5,
        9.3745188e4,
        6.5498167e4,
        2.9905138e4,
        1.1862497e4,
        3.2077308e3,
        1.2143969e3,
        4.5835989e2,
        2.3489897e+02,
        9.8558083e+01,
        5.1137517e+01,
        2.7608834e+01,
        2.0798428e+01,
        1.6845337e+01,
        1.6274940e+01,
        1.2408498e+01,
        1.1911952e+01,
        1.1497323e+01,
        8.9459829e+00,
        7.0864168e+00,
        6.7301243e+00,
        5.7713490e+00,
        4.7302819e+00,
        4.6333761e+00,
        4.2933964e+00,
        3.7107449e+00,
        3.6730447e+00,
        3.4306134e+00,
        3.0970636e+00,
        2.9112693e+00,
        2.8505157e+00,
        2.6257576e+00,
        2.4401424e+00,
        2.1292169e+00,
        2.0290664e+00,
        1.9779173e+00,
        1.8768543e+00,
        1.7682711e+00,
        1.6070525e+00,
        1.4763070e+00,
        1.4672250e+00,
        1.3749969e+00,
        1.3024763e+00,
        1.2374689e+00,
        1.1289259e+00,
        1.0790433e+00,
        1.0521576e+00,
        9.9653888e-01,
        9.5131321e-01,
        8.5311568e-01,
        8.2102341e-01,
        7.6819505e-01,
        6.9937573e-01,
        6.8601031e-01,
        6.8173644e-01,
        6.5468762e-01,
        6.0041885e-01,
        5.8021491e-01,
        5.4683827e-01,
        5.3010289e-01,
        5.1129752e-01,
        4.4822095e-01,
        4.1881901e-01,
        3.9664852e-01,
        3.7767897e-01,
        3.6862223e-01,
        3.4232498e-01,
        3.3763276e-01,
        3.1367506e-01,
        3.0996277e-01,
        2.7805110e-01
    ]
    iterations_for_costovertime = 0:5:200

    cost_over_time_uandv_10days = [2.2183104e+05,
        1.9025360e+05,
        1.2802771e+05,
        4.6342355e+04,
        2.0704170e+04,
        5.7256078e+03,
        2.9687462e+03,
        1.1620300e+03,
        5.0680867e+02,
        2.1361523e+02,
        1.0766945e+02,
        6.9604122e+01,
        6.5892764e+01,
        6.5200882e+01,
        5.0383360e+01,
        4.5268362e+01,
        4.0733260e+01,
        3.9674711e+01,
        3.3978593e+01,
        3.2567157e+01,
        2.8865247e+01,
        2.5375712e+01,
        2.4467403e+01,
        2.3286585e+01,
        2.1757497e+01,
        2.0532825e+01,
        2.0183028e+01,
        1.9858754e+01,
        1.8335790e+01,
        1.7202785e+01,
        1.6582841e+01,
        1.4503482e+01,
        1.4081102e+01,
        1.3723128e+01,
        1.2323489e+01,
        1.1659547e+01,
        1.0770358e+01,
        1.0648680e+01,
        1.0215325e+01,
        9.7986641e+00,
        9.6804921e+00,
        8.5572324e+00,
        8.4202218e+00,
        8.3640266e+00,
        8.1687442e+00,
        7.8443146e+00,
        7.7855382e+00,
        7.6774213e+00
    ]

    iterations = 0:5:235

end