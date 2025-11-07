import Pkg 
Pkg.activate("SpeedyWeather")

using SpeedyWeather, RingGrids, CairoMakie, GeoMakie, JLD2, Interpolations, LaTeXStrings

spectral_grid = SpectralGrid(trunc=32, nlayers=8)          # define resolution
model = PrimitiveWetModel(; spectral_grid) 

data_folder = "SpeedyWeather/data/"
d9 = JLD2.load(string(data_folder, "sensitivity-9temp.jld2"))["d_progn"]
d72 = JLD2.load(string(data_folder, "sensitivity-72temp.jld2"))["d_progn"]

ic9 = JLD2.load(string(data_folder, "sensitivity-9temp-ic.jld2"))["diagn"]
ic72 = JLD2.load(string(data_folder, "sensitivity-72temp-ic.jld2"))["diagn"]

function get_full_field(x::LowerTriangularArray)
    reduced_field = transform(x, model.spectral_transform)
    return get_full_field(reduced_field)
end

function get_full_field(x::Field)
    return RingGrids.interpolate(RingGrids.full_grid_type(x.grid), x.grid.nlat_half, x)
end

shift_and_flip(field) = circshift(Matrix(field), (RingGrids.get_nlon(typeof(field.grid), field.grid.nlat_half) ÷ 2, 0))[:, end:-1:1]

prepare_sensitivity_data(x) = shift_and_flip(get_full_field(x))

"""
    prepare_wind_field(u_grid, v_grid)

Convert u and v wind components from grid space to a matrix of Point2f for streamplot.
Applies get_full_field and shift_and_flip transformations to both components.

Returns a matrix of Point2f where each point contains (u, v) wind components.
"""
function prepare_wind_field(u_grid, v_grid)
    # Transform to full grid and shift/flip
    u_field = get_full_field(u_grid)
    v_field = get_full_field(v_grid)
    
    u_data = shift_and_flip(u_field)
    v_data = shift_and_flip(v_field)
    
    # Combine into matrix of Point2f
    nlon, nlat = size(u_data)
    wind_field = Matrix{Point2f}(undef, nlon, nlat)
    
    for i in 1:nlon, j in 1:nlat
        wind_field[i, j] = Point2f(u_data[i, j], v_data[i, j])
    end
    
    return wind_field
end

# get all the data we want 
all_data = []
all_winddata = []

# for vorticity senstivity data, we need to multiply by R to undo the scaling because the derivaitive is a temperature / vorticity ratio 
# 6h 

push!(all_data, prepare_sensitivity_data(d9.temp[:,8,2]))
push!(all_winddata, prepare_wind_field(ic9.grid.u_grid[:,8], ic9.grid.v_grid[:,8]))

push!(all_data, prepare_sensitivity_data(d9.vor[:,8,2] .* model.planet.radius))
push!(all_winddata, prepare_wind_field(ic9.grid.u_grid[:,8], ic9.grid.v_grid[:,8]))

push!(all_data, prepare_sensitivity_data(d9.vor[:,5,2] .* model.planet.radius))
push!(all_winddata, prepare_wind_field(ic9.grid.u_grid[:,5], ic9.grid.v_grid[:,5]))

push!(all_data, prepare_sensitivity_data(d9.pres[:,2]))
push!(all_winddata, prepare_wind_field(ic9.grid.u_grid[:,8], ic9.grid.v_grid[:,8]))

# 2d 

push!(all_data, prepare_sensitivity_data(d72.temp[:,8,2]))
push!(all_winddata, prepare_wind_field(ic72.grid.u_grid[:,8], ic72.grid.v_grid[:,8]))

push!(all_data, prepare_sensitivity_data(d72.vor[:,8,2] .* model.planet.radius))
push!(all_winddata, prepare_wind_field(ic72.grid.u_grid[:,8], ic72.grid.v_grid[:,8]))

push!(all_data, prepare_sensitivity_data(d72.vor[:,5,2] .* model.planet.radius))
push!(all_winddata, prepare_wind_field(ic72.grid.u_grid[:,5], ic72.grid.v_grid[:,5]))

push!(all_data, prepare_sensitivity_data(d72.pres[:,2]))
push!(all_winddata, prepare_wind_field(ic72.grid.u_grid[:,8], ic72.grid.v_grid[:,8]))

# Calculate global color range across all vorticity data 
cmax_limits(x) = 0.7*maximum(abs.(extrema(x)))
cmax_global_vorticity = cmax_limits(vcat(all_data[2], all_data[3], all_data[6], all_data[7]))
cmax_global_temp = cmax_limits(vcat(all_data[1], all_data[5]))
cmax_global_pres = cmax_limits(vcat(all_data[4], all_data[8]))

cmax_globals = (cmax_global_temp, cmax_global_vorticity, cmax_global_vorticity, cmax_global_pres, cmax_global_temp, cmax_global_vorticity, cmax_global_vorticity, cmax_global_pres)

# Titles 
titles = ["Temperature Layer 8; 6h", "Vorticity Layer 8; 6h", "Vorticity Layer 5; 6h", "Surface Pressure; 6h", "Temperature Layer 8; 2d", "Vorticity Layer 8; 2d", "Vorticity Layer 5; 2d", "Surface Pressure; 2d"]
#labels = ["∂T_f/∂T₀ []", "∂T_f/∂ζ₀ [Ks]", "∂T_f/∂ζ₀ [Ks]", "∂T_f/∂lnp₀ [K/lnPa]"]
labels = [L"\frac{\partial T_f}{\partial T_0}\quad[]", L"\frac{\partial T_f}{\partial\zeta_0}\quad[Ks]", L"\frac{\partial T_f}{\partial\zeta_0}\quad[Ks]", L"\frac{\partial T_f}{\partial\ln p_0}\quad[K/\ln Pa]", L"\frac{\partial T_f}{\partial T_0}\quad[]", L"\frac{\partial T_f}{\partial\zeta_0}\quad[Ks]", L"\frac{\partial T_f}{\partial\zeta_0}\quad[Ks]", L"\frac{\partial T_f}{\partial\ln p_0}\quad[K/\ln Pa]"]

# Create figure with 8 subplots (2 rows x 4 columns)
fig = Figure(size=(2400, 1200), fontsize = 35)

# Location to mark: 55°N, 11°E (Copenhagen area)
marker_lon = 11.0
marker_lat = 55.0

## PORTRAIT LAYOUT 

# Create subplots for each level
nlayers = length(all_data)
for k in 1:nlayers
    row = 2 * ((k - 1) ÷ 4) + 1 # Row index for GeoAxis (1, 3, 5, ...)
    col = (k - 1) % 4 + 1          # Column index (1-4)

    # Create GeoAxis
    ga = GeoAxis(
        fig[row, col];
        dest = "+proj=wintri",  # Winkel Tripel projection
        limits = ((-90, 90), (0, 80)),
        title = titles[k]
    )
    hidedecorations!(ga)
    
    # Get data for this level
    data = all_data[k]
    
    # Plot with meshimage
    sp = meshimage!(ga, -180..180, -90..90, data; 
        colormap = :bwr,
        colorrange = (-cmax_globals[k], cmax_globals[k])
    )
    
    # Add wind streamlines
    wind_data = all_winddata[k]
    xs = range(-180, 180, length=size(wind_data, 1))
    ys = range(-90, 90, length=size(wind_data, 2))
    wind_itp = LinearInterpolation((xs, ys), wind_data)
    streamplot!(ga, x -> wind_itp(x...), -180..180, -90..90; 
        arrow_size = 8, alpha=0.25)

    # Add coastlines
    lines!(ga, GeoMakie.coastlines(); color = :black, linewidth = 0.5)
    
    # Add cross marker at 55°N, 11°E
    scatter!(ga, [marker_lon], [marker_lat]; 
        marker = :cross, 
        markersize = 25, 
        color = :black, 
        strokewidth = 0.8)
    
    # Add label box (a-h) in upper left corner using Box
    panel_label = Char('a' + k - 1)  # Convert to 'a', 'b', 'c', etc.
    Label(fig[row, col], string(panel_label);
        tellwidth = false,
        tellheight = false,
        halign = :left,
        valign = :center,
        fontsize = 28,
        font = :bold,
        color = :black)
    
    # Add colorbar below this subplot
    Colorbar(fig[row + 1, col]; 
        limits = (-cmax_globals[k], cmax_globals[k]),
        colormap = :bwr,
        label = labels[k],
        vertical = false,
        flipaxis = false, 
        labelsize=25,
        ticklabelsize=25)
end

# tighter layout
rowgap!(fig.layout, 1, -270)
rowgap!(fig.layout, 2, -250)
rowgap!(fig.layout, 3, -270)

fig

save("speedyweather.png", fig)

## TRANSPOSED LAYOUT 

# Create figure with 8 subplots (4 rows x 2 columns)
fig = Figure(size=(1200, 2000), fontsize = 35)

# Location to mark: 55°N, 11°E (Copenhagen area)
marker_lon = 11.0
marker_lat = 55.0

# Create subplots for each level
nlayers = length(all_data)
for k in 1:nlayers
    row = 2 * ((k - 1) ÷ 2) + 1 # Row index for GeoAxis (1, 3, 5, ...)
    col = (k - 1) % 2 + 1          # Column index (1-4)

    # Create GeoAxis
    ga = GeoAxis(
        fig[row, col];
        dest = "+proj=wintri",  # Winkel Tripel projection
        limits = ((-90, 90), (0, 80)),
        title = titles[k]
    )
    hidedecorations!(ga)
    
    # Get data for this level
    data = all_data[k]
    
    # Plot with meshimage
    sp = meshimage!(ga, -180..180, -90..90, data; 
        colormap = :bwr,
        colorrange = (-cmax_globals[k], cmax_globals[k])
    )
    
    # Add wind streamlines
    wind_data = all_winddata[k]
    xs = range(-180, 180, length=size(wind_data, 1))
    ys = range(-90, 90, length=size(wind_data, 2))
    wind_itp = LinearInterpolation((xs, ys), wind_data)
    streamplot!(ga, x -> wind_itp(x...), -180..180, -90..90; 
        arrow_size = 8, alpha=0.25)

    # Add coastlines
    lines!(ga, GeoMakie.coastlines(); color = :black, linewidth = 0.5)
    
    # Add cross marker at 55°N, 11°E
    scatter!(ga, [marker_lon], [marker_lat]; 
        marker = :cross, 
        markersize = 25, 
        color = :black, 
        strokewidth = 0.8)
    
    # Add label box (a-h) in upper left corner using Box
    panel_label = Char('a' + k - 1)  # Convert to 'a', 'b', 'c', etc.
    Label(fig[row, col], string(panel_label);
        tellwidth = false,
        tellheight = false,
        halign = :left,
        valign = :center,
        fontsize = 28,
        font = :bold,
        color = :black)
    
    # Add colorbar below this subplot
    Colorbar(fig[row + 1, col]; 
        limits = (-cmax_globals[k], cmax_globals[k]),
        colormap = :bwr,
        label = labels[k],
        vertical = false,
        flipaxis = false, 
        labelsize=25,
        ticklabelsize=25)
end

# tighter layout
rowgap!(fig.layout, -100)

fig

save("speedyweather-transposed.pdf", fig)