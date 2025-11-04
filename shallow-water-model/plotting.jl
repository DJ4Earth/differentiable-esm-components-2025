include("technical_paper.jl")
using Parameters

# energy loss function

chkp = load_object("results/primal_10day_energyloss_100625.jld2")
dchkp = load_object("results/deriv_10day_energyloss_100625.jld2")

states = ShallowWaters.PrognosticVars{Float64}(ShallowWaters.remove_halo(
            chkp.S.Prog.u,
            chkp.S.Prog.v,
            chkp.S.Prog.η,
            chkp.S.Prog.sst,
            chkp.S
        )...)
dstates = ShallowWaters.PrognosticVars{Float64}(ShallowWaters.remove_halo(
            dchkp.S.Prog.u,
            dchkp.S.Prog.v,
            dchkp.S.Prog.η,
            dchkp.S.Prog.sst,
            chkp.S
        )...)

t = 224*5
fig = Figure(size=(600, 500));
ax1, hm1 = heatmap(fig[1,1], states.u[:, 1:end-1].^2 .+ states.v[1:end-1, :].^2,
colormap=:amp,
axis=(xlabel=L"x", ylabel=L"y", title=L"\mathcal{E}"),
colorrange=(0,
maximum(states.u[:, 1:end-1].^2 .+ states.v[1:end-1, :].^2))
);

Colorbar(fig[1,2], hm1)
ax1, hm1 = heatmap(fig[2,1], dstates.u,
colormap=:balance,
axis=(xlabel=L"x", ylabel=L"y", title=L"\partial \mathcal{E} / \partial u"),
colorrange=(-maximum(dstates.u),
maximum(dstates.u))
);

# derivative wrt wind stress
fig = Figure(fontsize = 15);
ax1, hm1 = heatmap(fig[1,1], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
dchkp.S.forcing.Fx,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\partial J / F_x"),
colorrange=(-maximum(dchkp.S.forcing.Fx),
maximum(dchkp.S.forcing.Fx))
);
Colorbar(fig[1,2], hm1)

# derivative wrt initial cond
fig = Figure(size=(900, 350), fontsize=15);
ax1, hm1 = heatmap(fig[1,1], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
dstates.u,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\partial J / u(t_0)"),
colorrange=(-maximum(dstates.u),
maximum(dstates.u))
);
Colorbar(fig[1,2], hm1)

ax2, hm2 = heatmap(fig[1,3], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
dstates.v,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\partial J / v(t_0)"),
colorrange=(-maximum(dstates.v),
maximum(dstates.v))
);
Colorbar(fig[1,4], hm2)

# time-averaged eta plot
eta = ncread("./128_10yearspinup_fromrest_noslipbc_epsetup/eta.nc", "eta")

S = ShallowWaters.model_setup(output=false,
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

@unpack Δ,x_u,y_u,Lx,Ly = S.grid
@unpack Fx0,Fy0,H,ρ,scale = S.parameters

eta_avg = zeros(128,128)
for t = 1:1123

    eta_avg += eta[:,:,t]

end
eta_avg = eta_avg / (1123)

fig = Figure(size = (800,500), fontsize = 15);
ax1, hm = heatmap(fig[1,1],
    LinRange(0, 3840, 128),
    LinRange(0, 3840, 128),
    eta_avg,
    colormap=:balance,
    axis=(xlabel="km",
    ylabel="km",
    title="Time-averaged sea surface height")
)
Colorbar(fig[1,2], label=L"m", colormap=:balance, colorrange=(-maximum(eta_avg),maximum(eta_avg)))

xx_u,yy_u = ShallowWaters.meshgrid(x_u,y_u)
Fx = (scale*Δ*Fx0/ρ/H)*(cos.(2π*(yy_u/Ly .- 1/2)) + 2*sin.(π*(yy_u/Ly .- 1/2)))

ax2 = Axis(fig[1,3], title="Wind-stress", xlabel="Pa")
lines!(ax2, Fx[1, :], yy_u[1,:])
hideydecorations!(ax2, ticks = false)

colsize!(fig.layout, 1, Auto(2))
# linkyaxes!(ax1, ax2)


######


# data assimilation

result = load_object("./results/result_10day_point1sigma_200iterations.jld2")

udata = ncread("./128_postspinup_1year_noslipbc_epsetup/u.nc", "u")
vdata = ncread("./128_postspinup_1year_noslipbc_epsetup/v.nc", "v")
etadata = ncread("./128_postspinup_1year_noslipbc_epsetup/eta.nc", "eta")
Prog_true, Prog_pred, Prog_nlp, upert, vpert, etapert, Prog_true10, Prog_pred10, Prog_nlp10 = ignore(result)

cost_pred = load_object("./costovertime_predictionmodel.jld2")
cost_opt = load_object("./costovertime_optimized.jld2")

# u and v fields before integrating
fig = Figure(fontsize = 15, size=(1000,650));

ax1, hm1 = heatmap(fig[1,1], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
udata[:,:,1],
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"u(t_0,x,y)"),
colorrange=(-maximum(Prog_true.u),
maximum(Prog_true.u))
);
Colorbar(fig[1,2], hm1, label="m/s")

ax3, hm3 = heatmap(fig[1,3], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_pred.u,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\tilde{u}(t_0,x,y)"),
colorrange=(-maximum(Prog_true.u),
maximum(Prog_true.u))
);
Colorbar(fig[1,4], hm3, label="m/s")

ax2, hm2 = heatmap(fig[1,5], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_nlp.u,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\tilde{u}(t_0, x, y, +)"),
colorrange=(-maximum(Prog_true.u),
maximum(Prog_true.u))
);
Colorbar(fig[1,6], hm2, label="m/s")

ax1, hm1 = heatmap(fig[2,1], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
vdata[:,:,1],
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"v(t_0,x,y)"),
colorrange=(-maximum(Prog_true.v),
maximum(Prog_true.v))
);
Colorbar(fig[2,2], hm1, label="m/s")

ax3, hm3 = heatmap(fig[2,3], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_pred.v,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\tilde{v}(t_0,x,y)"),
colorrange=(-maximum(Prog_true.v),
maximum(Prog_true.v))
);
Colorbar(fig[2,4], hm3,label="m/s")

ax2, hm2 = heatmap(fig[2,5], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_nlp.v,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\tilde{v}(t_0,x,y,+)"),
colorrange=(-maximum(Prog_true.v),
maximum(Prog_true.v))
);
Colorbar(fig[2,6], hm2, label="m/s")

# u and v fields after ten day integration

fig = Figure(fontsize = 15, size=(1000,550));

ax1, hm1 = heatmap(fig[1,1], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
udata[:,:,11],
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"u(t_f,x,y)"),
colorrange=(-maximum(abs.(udata[:,:,11])),
maximum(abs.(udata[:,:,11])))
);
Colorbar(fig[1,2], hm1, label="m/s")

ax3, hm3 = heatmap(fig[1,3], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_pred10.u,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\tilde{u}(t_f,x,y)"),
colorrange=(-maximum(abs.(udata[:,:,11])),
maximum(abs.(udata[:,:,11])))
);
Colorbar(fig[1,4], hm3, label="m/s")

ax2, hm2 = heatmap(fig[1,5], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_nlp10.u,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\tilde{u}(t_f,x,y,+)"),
colorrange=(-maximum(abs.(udata[:,:,11])),
maximum(abs.(udata[:,:,11])))
);
Colorbar(fig[1,6], hm2,label="m/s")

ax1, hm1 = heatmap(fig[2,1], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
vdata[:,:,11],
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"v(t_f,x,y)"),
colorrange=(-maximum(abs.(vdata[:,:,11])),
maximum(abs.(vdata[:,:,11])))
);
Colorbar(fig[2,2], hm1,label="m/s")

ax3, hm3 = heatmap(fig[2,3], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_pred10.v,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\tilde{v}(t_f,x,y)"),
colorrange=(-maximum(abs.(vdata[:,:,11])),
maximum(abs.(vdata[:,:,11])))
);
Colorbar(fig[2,4], hm3,label="m/s")

ax2, hm2 = heatmap(fig[2,5], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_nlp10.v,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\tilde{v}(t_f,x,y,+)"),
colorrange=(-maximum(abs.(vdata[:,:,11])),
maximum(abs.(vdata[:,:,11])))
);
Colorbar(fig[2,6], hm2,label="m/s")

ga = fig[1, 1] = GridLayout()
gb = fig[1, 3] = GridLayout()
gc = fig[1, 5] = GridLayout()
gd = fig[2, 1] = GridLayout()
ge = fig[2,3] = GridLayout()
gf = fig[2,5] = GridLayout()
for (label, layout) in zip(["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"], [ga, gb, gc, gd, ge, gf])
Label(layout[1, 1, TopLeft()], label,
    fontsize = 15,
    font = :bold,
    padding = (0, 5, 5, 0),
    halign = :right)
end

# energy before and after data assimilation, ten day integration

# utrue = ncread("./128_postspinup_1year_noslipbc_epsetup/u.nc", "u")[:,:,11]
# vtrue = ncread("./128_postspinup_1year_noslipbc_epsetup/v.nc", "v")[:,:,11]

fig = Figure(size=(1000, 300));
ax1, hm1 = heatmap(fig[1,1], LinRange(0, 3840, 128),
    LinRange(0, 3840, 128),
    udata[:, 1:end-1,11].^2 .+ vdata[1:end-1, :, 11].^2,
    colormap=:amp,
    axis=(xlabel="km", ylabel="km", title="True energy after 10 days"),
    colorrange=(0,
    maximum(Prog_true10.u[:, 1:end-1].^2 .+ Prog_true10.v[1:end-1, :].^2))
)
Colorbar(fig[1,2], hm1)

ax1, hm1 = heatmap(fig[1,3], LinRange(0, 3840, 128),
    LinRange(0, 3840, 128),
    Prog_pred10.u[:, 1:end-1].^2 .+ Prog_pred10.v[1:end-1, :].^2,
    colormap=:amp,
    axis=(xlabel="km", ylabel="km", title="Predicted energy after 10 days"),
    colorrange=(0,
    maximum(Prog_true10.u[:, 1:end-1].^2 .+ Prog_true10.v[1:end-1, :].^2))
)
Colorbar(fig[1,4], hm1)

ax1, hm1 = heatmap(fig[1,5], LinRange(0, 3840, 128),
    LinRange(0, 3840, 128),
    Prog_nlp10.u[:, 1:end-1].^2 .+ Prog_nlp10.v[1:end-1, :].^2,
    colormap=:amp,
    axis=(xlabel="km", ylabel="km", title="Inferred energy after 10 days"),
    colorrange=(0,
    maximum(Prog_true10.u[:, 1:end-1].^2 .+ Prog_true10.v[1:end-1, :].^2))
)
Colorbar(fig[1,6], hm1)

ga = fig[1, 1] = GridLayout()
gb = fig[1, 3] = GridLayout()
gc = fig[1, 5] = GridLayout()
for (label, layout) in zip(["(a)", "(b)", "(c)"], [ga, gb, gc])
Label(layout[1, 1, TopLeft()], label,
    fontsize = 15,
    font = :bold,
    padding = (0, 5, 5, 0),
    halign = :right)
end

# ax1, hm1 = heatmap(fig[2,1], LinRange(0, 3840, 128),
#     LinRange(0, 3840, 128),
#     Prog_true.u[:, 1:end-1].^2 .+ Prog_true.v[1:end-1, :].^2,
#     colormap=:amp,
#     axis=(xlabel="km", ylabel="km", title="True energy after 10 days"),
#     colorrange=(0,
#     maximum(Prog_true10.u[:, 1:end-1].^2 .+ Prog_true10.v[1:end-1, :].^2))
# )
# Colorbar(fig[2,2], hm1)

# ax1, hm1 = heatmap(fig[2,3], LinRange(0, 3840, 128),
#     LinRange(0, 3840, 128),
#     Prog_pred.u[:, 1:end-1].^2 .+ Prog_pred.v[1:end-1, :].^2,
#     colormap=:amp,
#     axis=(xlabel="km", ylabel="km", title="Predicted energy after 10 days"),
#     colorrange=(0,
#     maximum(Prog_true10.u[:, 1:end-1].^2 .+ Prog_true10.v[1:end-1, :].^2))
# )
# Colorbar(fig[2,4], hm1)

# ax1, hm1 = heatmap(fig[2,5], LinRange(0, 3840, 128),
#     LinRange(0, 3840, 128),
#     Prog_nlp.u[:, 1:end-1].^2 .+ Prog_nlp.v[1:end-1, :].^2,
#     colormap=:amp,
#     axis=(xlabel="km", ylabel="km", title="Adjoint energy after 10 days"),
#     colorrange=(0,
#     maximum(Prog_true10.u[:, 1:end-1].^2 .+ Prog_true10.v[1:end-1, :].^2))
# )
# Colorbar(fig[2,6], hm1)


# difference in u
fig = Figure(fontsize = 20, size=(900,500));

# ax1, hm1 = heatmap(fig[1,1], LinRange(0, 3840, 128),
# LinRange(0, 3840, 128),
# Prog_true.u .- Prog_pred.u,
# colormap=:balance,
# axis=(xlabel="km", ylabel="km", title="Perturbation")
# # colorrange=(-maximum(Prog_true.u .- Prog_nlp.u), maximum(Prog_true.u .- Prog_nlp.u))
# );
# Colorbar(fig[1,2], hm1)

ax1, hm1 = heatmap(fig[1,1], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
upert,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title="Perturbation"),
colorrange=(-maximum(upert), maximum(upert))
);
Colorbar(fig[1,2], hm1)

ax2, hm2 = heatmap(fig[1,3], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
abs.(Prog_true.u .- Prog_nlp.u),
colormap=:amp,
axis=(xlabel="km", ylabel="km", title=L"|u(t_0) - \tilde{u}(t_0)|"),
colorrange=(0,
maximum(abs.(Prog_true.u .- Prog_nlp.u)))
);
Colorbar(fig[1,4], hm1)


# v field
fig = Figure(fontsize = 15, size=(900,800));

ax1, hm1 = heatmap(fig[1,1], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_true.v,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"u(t_0)"),
colorrange=(-maximum(Prog_true.v),
maximum(Prog_true.v))
);
Colorbar(fig[1,2], hm1)

ax2, hm2 = heatmap(fig[1,3], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_nlp.v,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\tilde{v(t_0, +)}"),
colorrange=(-maximum(Prog_true.v),
maximum(Prog_true.v))
);
Colorbar(fig[1,4], hm2)

ax3, hm3 = heatmap(fig[2,1], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_pred.v,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\tilde{v(t_0, -)}"),
colorrange=(-maximum(Prog_true.v),
maximum(Prog_true.v))
);
Colorbar(fig[2,2], hm3)

# η plots
fig = Figure(fontsize = 20, size=(900,500));

ax1, hm1 = heatmap(fig[1,1], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_true.η,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\eta(t_0)"),
# colorrange=(-maximum(Prog_pred.η), maximum(Prog_pred.η))
);
Colorbar(fig[1,2], hm1)

ax1, hm1 = heatmap(fig[1,3], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_pred.η,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\tilde{\eta}(t_0"),
# colorrange=(-maximum(Prog_pred.η), maximum(Prog_pred.η))
);
Colorbar(fig[1,4], hm1)

# ax1, hm1 = heatmap(fig[1,1], LinRange(0, 3840, 128),
# LinRange(0, 3840, 128),
# Prog_true.u .- Prog_pred.u,
# colormap=:balance,
# axis=(xlabel="km", ylabel="km", title="Perturbation")
# # colorrange=(-maximum(Prog_true.u .- Prog_nlp.u), maximum(Prog_true.u .- Prog_nlp.u))
# );
# Colorbar(fig[1,2], hm1)

# difference plots next to initial perturbations for all three prognostic variables
fig = Figure(size=(800, 700), fontsize=15);
ax1, hm1 = heatmap(fig[1,1], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
upert,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\text{Initial perturbation in }u(t_0)"),
colorrange=(-maximum(abs.(upert)), maximum(abs.(upert)))
);
Colorbar(fig[1,2], hm1, label="m/s")

ax2, hm2 = heatmap(fig[2,1], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_true.u .- Prog_nlp.u,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"u(t_0) - \tilde{u}(t_0)"),
colorrange=(-maximum(abs.(upert)), maximum(abs.(upert)))
);
Colorbar(fig[2,2], hm2, label="m/s")

ax3, hm3 = heatmap(fig[1,3], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
vpert,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\text{Initial perturbation in }v(t_0)"),
colorrange=(-maximum(abs.(vpert)), maximum(abs.(vpert)))
);
Colorbar(fig[1,4], hm3, label="m/s")

ax4, hm4 = heatmap(fig[2,3], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_true.v .- Prog_nlp.v,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"v(t_0) - \tilde{v}(t_0)"),
colorrange=(-maximum(abs.(vpert)), maximum(abs.(vpert)))
);
Colorbar(fig[2,4], hm4, label="m/s")

ga = fig[1, 1] = GridLayout()
gb = fig[1, 3] = GridLayout()
gc = fig[2, 1] = GridLayout()
gd = fig[2, 3] = GridLayout()
for (label, layout) in zip(["(a)", "(b)", "(c)", "(d)"], [ga, gb, gc, gd])
Label(layout[1, 1, TopLeft()], label,
    fontsize = 15,
    font = :bold,
    padding = (0, 5, 5, 0),
    halign = :right)
end

# initial conditions and final states for predicted and adjoint runs

fig = Figure(size=(800, 700), fontsize=15);
ax1, hm1 = heatmap(fig[1,1], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_pred.u,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\tilde{u}(t_0, -)"),
colorrange=(-maximum(abs.(Prog_true.u)), maximum(abs.(Prog_true.u)))
);
Colorbar(fig[1,2], hm1, label="m/s")

ax2, hm2 = heatmap(fig[1,3], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_pred10.u,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\tilde{u}(t_f, -)"),
colorrange=(-maximum(abs.(Prog_true10.u)), maximum(abs.(Prog_true10.u)))
);
Colorbar(fig[1,4], hm2, label="m/s")

ax3, hm3 = heatmap(fig[2,1], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_nlp.u,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\tilde{u}(t_0)"),
colorrange=(-maximum(abs.(Prog_true.u)), maximum(abs.(Prog_true.u)))
);
Colorbar(fig[2,2], hm3, label="m/s")

ax4, hm4 = heatmap(fig[2,3], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_nlp10.u,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\tilde{u}(t_f)"),
colorrange=(-maximum(abs.(Prog_true10.u)), maximum(abs.(Prog_true10.u)))
);
Colorbar(fig[2,4], hm4, label="m/s")

ga = fig[1, 1] = GridLayout()
gb = fig[1, 3] = GridLayout()
gc = fig[2, 1] = GridLayout()
gd = fig[2, 3] = GridLayout()
for (label, layout) in zip(["(a)", "(b)", "(c)", "(d)"], [ga, gb, gc, gd])
Label(layout[1, 1, TopLeft()], label,
    fontsize = 15,
    font = :bold,
    padding = (0, 5, 5, 0),
    halign = :right)
end

# initial conditions and final state differences for predicted and adjoint runs

fig = Figure(size=(800, 700), fontsize=15);
ax1, hm1 = heatmap(fig[1,1], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_pred.u,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\tilde{u}(t_0, -)"),
colorrange=(-maximum(abs.(Prog_true.u)), maximum(abs.(Prog_true.u)))
);
Colorbar(fig[1,2], hm1, label="m/s")

ax2, hm2 = heatmap(fig[1,3], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_true10.u .- Prog_pred10.u,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\tilde{u}(t_f, -)"),
colorrange=(-maximum(abs.(Prog_true10.u)), maximum(abs.(Prog_true10.u)))
);
Colorbar(fig[1,4], hm2, label="m/s")

ax3, hm3 = heatmap(fig[2,1], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_nlp.u,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\tilde{u}(t_0)"),
colorrange=(-maximum(abs.(Prog_true.u)), maximum(abs.(Prog_true.u)))
);
Colorbar(fig[2,2], hm3, label="m/s")

ax4, hm4 = heatmap(fig[2,3], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_true10.u .- Prog_nlp10.u,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\tilde{u}(t_f)"),
colorrange=(-maximum(abs.(Prog_true10.u)), maximum(abs.(Prog_true10.u)))
);
Colorbar(fig[2,4], hm4, label="m/s")

ga = fig[1, 1] = GridLayout()
gb = fig[1, 3] = GridLayout()
gc = fig[2, 1] = GridLayout()
gd = fig[2, 3] = GridLayout()
for (label, layout) in zip(["(a)", "(b)", "(c)", "(d)"], [ga, gb, gc, gd])
Label(layout[1, 1, TopLeft()], label,
    fontsize = 15,
    font = :bold,
    padding = (0, 5, 5, 0),
    halign = :right)
end

# ax5, hm5 = heatmap(fig[1,5], LinRange(0, 3840, 128),
# LinRange(0, 3840, 128),
# etapert,
# colormap=:balance,
# axis=(xlabel="km", ylabel="km", title=L"\text{Initial perturbation in }\eta(t_0)"),
# colorrange=(-maximum(etapert), maximum(etapert))
# );
# Colorbar(fig[1,6], hm5)

# ax6, hm6 = heatmap(fig[2,5], LinRange(0, 3840, 128),
# LinRange(0, 3840, 128),
# Prog_true.η .- Prog_nlp.η,
# colormap=:balance,
# axis=(xlabel="km", ylabel="km", title=L"|\eta(t_0) - \tilde{\eta}(t_0)|"),
# colorrange=(-maximum(etapert), maximum(etapert))
# );
# Colorbar(fig[2,6], hm6)

# final states plus cost function

fig = Figure(fontsize = 15, size=(950, 700));

ax1, hm1 = heatmap(fig[1,1], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
udata[:,:,11],
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"u(x,y,t_f)"),
colorrange=(-maximum(abs.(udata[:,:,11])),
maximum(abs.(udata[:,:,11])))
);
Colorbar(fig[1,2], hm1, label="m/s")

ax3, hm3 = heatmap(fig[1,3], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_pred10.u,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\tilde{u}(x,y,t_f)"),
colorrange=(-maximum(abs.(udata[:,:,11])),
maximum(abs.(udata[:,:,11])))
);
Colorbar(fig[1,4], hm3, label="m/s")

ax2, hm2 = heatmap(fig[2,1], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_nlp10.u,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\tilde{u}(x,y,t_f,+)"),
colorrange=(-maximum(abs.(udata[:,:,11])),
maximum(abs.(udata[:,:,11])))
);
Colorbar(fig[2,2], hm2,label="m/s")

ax = Axis(fig[2,3],xlabel="Days",
        ylabel=L"J_t (m^2/s^2)",
        yscale=log10,
)
scatterlines!(ax, 1:10, cost_pred, label="With perturbed initial conditions")
scatterlines!(ax, 1:10, cost_opt, label="With optimized initial conditions")
axislegend(position=:rc)
# Legend(fig[2,4], ax)


ga = fig[1, 1] = GridLayout()
gb = fig[1, 3] = GridLayout()
gc = fig[2, 1] = GridLayout()
gd = fig[2, 3] = GridLayout()
for (label, layout) in zip(["(a)", "(b)", "(c)", "(d)"], [ga, gb, gc, gd])
Label(layout[1, 1, TopLeft()], label,
    fontsize = 15,
    font = :bold,
    padding = (0, 5, 5, 0),
    halign = :right)
end

# initial states plus cost function and perturbation

fig = Figure(fontsize = 15, size=(1400,550));

ax1, hm1 = heatmap(fig[1,1], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
upert,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title="Initial perturbation"),
colorrange=(-maximum(abs.(udata[:,:,1])),
maximum(abs.(udata[:,:,1])))
);
Colorbar(fig[1,2], hm1, label="m/s")

ax1, hm1 = heatmap(fig[1,3], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
udata[:,:,1],
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"u(t_0,x,y)"),
colorrange=(-maximum(abs.(udata[:,:,1])),
maximum(abs.(udata[:,:,1])))
);
Colorbar(fig[1,4], hm1, label="m/s")

ax3, hm3 = heatmap(fig[1,5], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_pred.u,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\tilde{u}(t_0,x,y)"),
colorrange=(-maximum(abs.(udata[:,:,1])),
maximum(abs.(udata[:,:,1])))
);
Colorbar(fig[1,6], hm3, label="m/s")

ax2, hm2 = heatmap(fig[1,7], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_nlp.u,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\tilde{u}(t_0,x,y,+)"),
colorrange=(-maximum(abs.(udata[:,:,1])),
maximum(abs.(udata[:,:,1])))
);
Colorbar(fig[1,8], hm2,label="m/s")

# ax = Axis(fig[2,:],xlabel="Iterations",
#         ylabel="Loss value (m/s)",
#         yscale=log10,
# )
# lines!(ax, iterations, cost_over_time_uandv_10days)


ga = fig[1, 1] = GridLayout()
gb = fig[1, 3] = GridLayout()
gc = fig[1, 5] = GridLayout()
gd = fig[1, 7] = GridLayout()
# gd = fig[2, 1] = GridLayout()
for (label, layout) in zip(["(a)", "(b)", "(c)", "(d)"], [ga, gb, gc, gd])
Label(layout[1, 1, TopLeft()], label,
    fontsize = 15,
    font = :bold,
    padding = (0, 5, 5, 0),
    halign = :right)
end

# another layout option for the above

fig = Figure(fontsize = 15, size=(700,550));

ax1, hm1 = heatmap(fig[1,1], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
upert,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title="Initial perturbation"),
colorrange=(-maximum(abs.(udata[:,:,1])),
maximum(abs.(udata[:,:,1])))
);
Colorbar(fig[1,2], hm1, label="m/s")

ax1, hm1 = heatmap(fig[1,3], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
udata[:,:,1],
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"u(x,y,t_0)"),
colorrange=(-maximum(abs.(udata[:,:,1])),
maximum(abs.(udata[:,:,1])))
);
Colorbar(fig[1,4], hm1, label="m/s")

ax3, hm3 = heatmap(fig[2,1], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_pred.u,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\tilde{u}(x,y,t_0)"),
colorrange=(-maximum(abs.(udata[:,:,1])),
maximum(abs.(udata[:,:,1])))
);
Colorbar(fig[2,2], hm3, label="m/s")

ax2, hm2 = heatmap(fig[2,3], LinRange(0, 3840, 128),
LinRange(0, 3840, 128),
Prog_nlp.u,
colormap=:balance,
axis=(xlabel="km", ylabel="km", title=L"\tilde{u}(x,y,t_0,+)"),
colorrange=(-maximum(abs.(udata[:,:,1])),
maximum(abs.(udata[:,:,1])))
);
Colorbar(fig[2,4], hm2,label="m/s")

ax = Axis(fig[3,:],xlabel="Iterations",
        ylabel="Loss value (m/s)",
        yscale=log10,
)
lines!(ax, iterations, cost_over_time_uandv_10days)


ga = fig[1, 1] = GridLayout()
gb = fig[1, 3] = GridLayout()
gc = fig[2, 1] = GridLayout()
gd = fig[2, 3] = GridLayout()
# ge = fig[3, 1] = GridLayout()
for (label, layout) in zip(["(a)", "(b)", "(c)", "(d)"], [ga, gb, gc, gd])
Label(layout[1, 1, TopLeft()], label,
    fontsize = 15,
    font = :bold,
    padding = (0, 5, 5, 0),
    halign = :right)
end


# Just the cost as a function of time
fig = Figure(fontsize=17, size=(800,300));
ax = Axis(fig[1,1],xlabel="Days",
        ylabel="J (m/s)",
        yscale=log10,
)
scatterlines!(ax, 1:10, cost_pred, label="Predicted initial state")
scatterlines!(ax, 1:10, cost_opt, label="Inferred initial state")
# axislegend(position=:rt)
Legend(fig[1,2], ax)