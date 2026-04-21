"""
Module for estimating annual property tax revenue from a data center of a given
size in Maine
"""
# Import packages
from pathlib import Path
import os
import json
import pickle
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.optimize import root_scalar

from bokeh.io import output_file
from bokeh.plotting import figure, show
from bokeh.palettes import Reds9
from bokeh.models import (
    GeoJSONDataSource, HoverTool, ColumnDataSource, Legend, LegendItem, Title
)


def make_me_elec_fmv_plot():
    """
    Create Bokeh scatter plot of electrical capacity (MW) and fair market value
    (FMV) along with linear regression line that estimates the realtionship.
    """
    # -------------------------------------------------------------------------
    # Load the data
    # -------------------------------------------------------------------------
    main_dir = Path(__file__).resolve().parent
    data_dir = os.path.join(main_dir, "data")
    images_dir = os.path.join(main_dir, "images")
    df = pd.read_excel(
        os.path.join(data_dir, "us_data_centers_FMV.xlsx"),
        sheet_name="dataset"
    )

    # Create fmv_billions variable
    df["fmv_billions"] = df["supplemental_public_value_usd"] / 1e9
    # For row 3 (Vantage VA31 (43330 Data Dr parcel), use the
    # assessed_full_market_value_usd variable
    df.loc[2, "fmv_billions"] = df.loc[
        2, "assessed_full_market_value_usd"
    ] / 1e9

    # Create a sample version of the DataFrame df_samp with only the columns
    # facility_name, operator, city, county, state,
    # electrical_capacity_mw_public, and fmv_billions
    df_samp = df[[
        "facility_name", "operator", "city", "county", "state",
        "electrical_capacity_mw_public", "fmv_billions", "status"
    ]].copy()
    print(df_samp.keys())
    print(df_samp.describe())

    # Remove three outliers with electrical capacity above 600 MW and FMV < $1B
    df_samp = df_samp.loc[
        ~(
            (df["electrical_capacity_mw_public"] > 600) &
            (df["fmv_billions"] < 1)
        )
    ].copy()

    # Remove any rows that have missing or NAN values in the
    # electrical_capacity_mw_public or fmv_billions.
    df_samp = df_samp.dropna(
        subset=["electrical_capacity_mw_public", "fmv_billions"]
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Estimate linear regression coefficients:
    # FMV = f(ElecCapac) = a_lin * ElecCapac + b_lin for ElecCapac > 95 MW
    #     = g(ElecCapac) = e^{a_exp * ElecCapac + b_exp} + c_exp
    #                      for ElecCapac <= 95 MW
    # where a_exp, b_exp, and c_exp are chosen such that
    # g(95) = f(95), g(0.9) = f(0.9), and g'(95) = f'(95)
    # -------------------------------------------------------------------------
    ElecCapacCutoff = 95
    ElecCapacLow = 0.9
    Xlin_mat = np.column_stack((
        np.ones(len(df_samp)),
        df_samp["electrical_capacity_mw_public"]
    ))
    yvec = df_samp["fmv_billions"].values
    ab_vec_lin = np.linalg.inv(Xlin_mat.T @ Xlin_mat) @ Xlin_mat.T @ yvec
    a_lin, b_lin = ab_vec_lin[1], ab_vec_lin[0]
    print("a_lin (slope coefficient):", a_lin)
    print("b_lin (intercept):", b_lin)

    # Start with fitting the 2-parameter exponential stitched function
    # g(x) = e^{a_exp1 * x} + c_exp1, such that g(95)=f(95) and g'(95)=f'(95).
    # This gives us a good initial guess for a_exp2 in our main specification.
    def a_exp_root_func1(a_exp, ElecCapacCutoff, a_lin):
        error = np.log(a_exp) + a_exp * ElecCapacCutoff - np.log(a_lin)
        return error
    sol1 = root_scalar(
        a_exp_root_func1, args=(ElecCapacCutoff, a_lin),
        bracket=[1e-9, 1e4], x0=0.5
    )
    a_exp1 = sol1.root
    c_exp1 = a_lin * ElecCapacCutoff + b_lin - np.exp(a_exp1 * ElecCapacCutoff)
    print("a_exp1 (exponential coefficient):", a_exp1)
    print("c_exp1 (additive constant):", c_exp1)

    # Write a root finder that solves for a_exp such that the following
    # equation holds: np.ln(a_exp) + a_exp * ElecCapacCutoff - np.ln(a_lin) = 0
    def b_exp_c_exp_func_a_exp(
        a_exp, ElecCapacCutoff, ElecCapacLow, a_lin, b_lin
    ):
        b_exp = np.log(a_lin) - np.log(a_exp) - a_exp * ElecCapacCutoff
        c_exp = (
            a_lin * ElecCapacCutoff + b_lin -
            np.exp(a_exp * ElecCapacCutoff + b_exp)
        )
        return b_exp, c_exp


    def a_exp_root_func2(a_exp, ElecCapacCutoff, ElecCapacLow, a_lin, b_lin):
        b_exp, c_exp = b_exp_c_exp_func_a_exp(
            a_exp, ElecCapacCutoff, ElecCapacLow, a_lin, b_lin
        )
        error = (
            np.exp(a_exp * ElecCapacLow + b_exp) + c_exp - 0.0007786
        )
        return error

    sol2 = root_scalar(
        a_exp_root_func2, args=(ElecCapacCutoff, ElecCapacLow, a_lin, b_lin),
        x0=a_exp1
    )
    a_exp2 = sol2.root
    print("a_exp2 (exponential coefficient):", a_exp2)
    b_exp2, c_exp2 = b_exp_c_exp_func_a_exp(
        a_exp2, ElecCapacCutoff, ElecCapacLow, a_lin, b_lin
    )
    print("b_exp2 (exponential intercept):", b_exp2)
    print("c_exp2 (additive constant):", c_exp2)
    print(dir(sol2))

    elec_capac_vec = np.linspace(
        df_samp["electrical_capacity_mw_public"].min(),
        df_samp["electrical_capacity_mw_public"].max(),
        500
    )
    elec_capac_vec_gtcut = elec_capac_vec[elec_capac_vec >= ElecCapacCutoff]
    elec_capac_vec_lecut = elec_capac_vec[elec_capac_vec <= ElecCapacCutoff]
    FMV_pred_lin = a_lin * elec_capac_vec_gtcut + b_lin
    FMV_pred_lin_lecut = a_lin * elec_capac_vec_lecut + b_lin
    # FMV_pred_exp1 = np.exp(a_exp1 * elec_capac_vec_lecut) + c_exp1
    FMV_pred_exp2 = np.exp(a_exp2 * elec_capac_vec_lecut + b_exp2) + c_exp2

    # -------------------------------------------------------------------------
    # Make figure
    # -------------------------------------------------------------------------
    fig1_title = (
        "Figure 1. US data centers by electrical capacity (MW) and fair " +
        "market value (FMV)"
    )
    # fig1_title = ""
    fig1_filename = "us_elec_fmv_plot.html"
    output_file(
        os.path.join(images_dir, fig1_filename),
        title=fig1_title, mode='inline'
    )

    TOOLS = "pan, box_zoom, wheel_zoom, save, reset, help"

    fig1 = figure(
        title=fig1_title,
        tools=TOOLS,
        x_axis_label="Electrical Capacity (MW)",
        y_axis_label="Fair Market Value ($Bil)",
        toolbar_location="right"
    )
    fig1.toolbar.logo = None
    source = ColumnDataSource(df_samp)
    scatter_renderer = fig1.scatter(
        x="electrical_capacity_mw_public",
        y="fmv_billions",
        source=source,
        size=10,
        color="green",
        alpha=0.5,
        hover_color="red",
        hover_alpha=1.0,
        hover_line_color="black",
        hover_line_width=1.5,
        legend_label="Data Centers"
    )
    fig1.line(
        x=elec_capac_vec_gtcut,
        y=FMV_pred_lin,
        line_width=2,
        color="blue",
        legend_label="Linear Fit"
    )
    fig1.line(
        x=elec_capac_vec_lecut,
        y=FMV_pred_lin_lecut,
        line_width=2,
        color="blue",
        line_dash="dashed",
        legend_label="Linear Fit (for <= 95 MW)"
    )
    # fig1.line(
    #     x=elec_capac_vec_lecut,
    #     y=FMV_pred_exp1,
    #     line_width=2,
    #     color="orange",
    #     legend_label="Exponential Fit 1 (for <= 95 MW)"
    # )
    fig1.line(
        x=elec_capac_vec_lecut,
        y=FMV_pred_exp2,
        line_width=2,
        color="red",
        legend_label="Exponential Fit (for <= 95 MW)"
    )
    hover = HoverTool(renderers=[scatter_renderer])
    hover.tooltips = [
        ("Facility Name", "@facility_name"),
        ("City", "@city"),
        ("County", "@county"),
        ("State", "@state"),
        ("Electrical Capacity (MW)", "@electrical_capacity_mw_public"),
        ("Fair Market Value (Billions USD)", "@fmv_billions{$0.000}"),
        ("Status", "@status")
    ]
    fig1.add_tools(hover)
    fig1.legend.location = "top_left"
    fig1.add_layout(fig1.legend[0], 'center')

    note_text_list1 = [
        (
            'Source: Richard W. Evans (@RickEcon), updated Apr. 20, 2026. ' +
            'Electrical capacity and fair market'
        ),
        (
            '    value data come from an individual search. The data in ' +
            'this figure with sources are available in the'
        ),
        (
            '    data/us_data_centers_FMV.xlsx file in the ' +
            'https://github.com/OpenSourceEcon/ME-DataCtrCostRev'
        ),
        (
            '    GitHub repository.'
        )
    ]


    for note_text in note_text_list1:
        caption1 = Title(
            text=note_text, align='left', text_font_size='9pt',
            text_font_style='italic',
            text_color='black',
            standoff=0
        )
        fig1.add_layout(caption1, 'below')

    show(fig1)


def elec_fmv_proptax_func(
    elec_capac: int | float = 100,
    tax_exempt_pct: int | float = 0.0,
    county: str = "Cumberland",
    print_out: bool = False
):
    # Set hard coded fair market value (FMV) function parameters
    a_lin = 0.010018082083149032
    b_lin = -0.3496020387036265
    a_exp = 0.010396604326045984
    b_exp = -1.0247649882273389
    c_exp = -0.36147598374031675
    ElecCapacCutoff = 95
    # Make sure elec_capac is a scalar within the range 0.9 to 2200 MW
    if elec_capac < 0.9 or elec_capac > 2200:
        raise ValueError(
            "ERROR elec_fmv_proptax_func(): elec_capac must be between 0.9 " +
            "and 2200 MW"
        )
    # Make sure tax_exempt_pct is between 0 and 1
    if tax_exempt_pct < 0 or tax_exempt_pct > 1:
        raise ValueError(
            "ERROR elec_fmv_proptax_func(): tax_exempt_pct must be between " +
            "0 and 1"
        )
    # Create proptax_rate_2024_dict dictionary that uses the
    # ME_FullValuePropTaxRates_cnty_2024.csv file in the data directory to map
    # each Maine county to its 2024 full value property tax rate
    # (proptax_rate_2024)
    # proptax_rate_2024_dict = pd.read_csv(
    #     os.path.join(
    #         Path(__file__).resolve().parent, "data",
    #         "ME_FullValuePropTaxRates_cnty_2024.csv"
    #     ),
    #     skiprows=2
    # ).dropna(
    #     subset=["county"]
    # ).set_index("county")["proptax_pct_2024"].to_dict()
    proptax_rate_2024_dict = {
        "Androscoggin": 0.01285,
        "Aroostook": 0.01596,
        "Cumberland": 0.01062,
        "Franklin": 0.00859,
        "Hancock": 0.00839,
        "Kennebec": 0.01107,
        "Knox": 0.01058,
        "Lincoln": 0.00753,
        "Oxford": 0.0094,
        "Penobscot": 0.0138,
        "Piscataquis": 0.01004,
        "Sagadahoc": 0.01035,
        "Somerset": 0.01185,
        "Waldo": 0.01137,
        "Washington": 0.01188,
        "York": 0.00876
    }
    # Make sure that the county string is one of the 16 Maine counties
    maine_counties = list(proptax_rate_2024_dict.keys())
    if county not in maine_counties:
        raise ValueError(
            "ERROR elec_fmv_proptax_func(): county must be one of the " +
            f"following: {', '.join(maine_counties)}"
        )
    # Compute the fair market value (FMV)
    if elec_capac >= ElecCapacCutoff:
        fmv = a_lin * elec_capac + b_lin
    else:
        fmv = np.exp(a_exp * elec_capac + b_exp) + c_exp
    # Calculate the annual property tax revenue for a data center with the
    # given electrical capacity, tax exeption percentage, and county.
    proptax_rate = proptax_rate_2024_dict[county]
    proptax_revenue = fmv * proptax_rate * (1 - tax_exempt_pct)
    if print_out:
        print(f"Electrical Capacity (MW): {elec_capac}")
        print(f"Fair Market Value: ${fmv*1e9:,.0f}")
        print(f"Property Tax Rate: {proptax_rate:.3%}")
        print(f"Tax Exemption Percentage: {tax_exempt_pct:.1%}")
        print(
            f"Annual Property Tax Revenue: ${proptax_revenue*1e9:,.0f}"
        )

    return float(fmv), proptax_rate, float(proptax_revenue)


def make_me_datactrcostrevmap(
    create_data=False, save_data=True
):
    """
    Create Bokeh map of main average county property tax rate
    """
    main_dir = Path(__file__).resolve().parent
    data_dir = os.path.join(main_dir, "data")
    images_dir = os.path.join(main_dir, "images")

    if create_data:
        print("Creating all the data from shapefiles.")
        start_time_all = time.time()
        # ---------------------------------------------------------------------
        # Add Maine state boundary shape file
        # ---------------------------------------------------------------------
        # Download U.S. states shape files from US Census Bureau
        # "https://www2.census.gov/geo/tiger/GENZ2023/shp/" +
        # "cb_2023_us_state_500k.zip"
        print("")
        print("Creating Maine state boundary shapefile,")
        start_time_me = time.time()
        us_shapefile_path = (
            os.path.join(
                data_dir, "shp", "cb_2023_us_state_500k",
                "cb_2023_us_state_500k.shp"
            )
        )
        states_gdf = gpd.GeoDataFrame.from_file(us_shapefile_path)
        states_gdf_json = states_gdf.to_json()
        states_gjson = json.loads(states_gdf_json)

        # Build a Maine polygon GeoDataFrame (not GeoJSON) for spatial ops
        me_gdf = states_gdf.loc[states_gdf["STUSPS"] == "ME"].copy()
        # Dissolve in case ME is multipart; makes a single boundary geometry
        me_gdf = me_gdf.dissolve()

        me_gdf_str = me_gdf.to_json()
        me_src = GeoJSONDataSource(geojson=me_gdf_str)

        elapsed_time_me = time.time() - start_time_me
        min = int(elapsed_time_me // 60)
        sec = np.round(elapsed_time_me % 60, 1)
        print(f"took {min} minutes and {sec} seconds.")

        # ---------------------------------------------------------------------
        # Add Maine county boundaries shape file
        # ---------------------------------------------------------------------
        print("")
        print("Creating county boundaries shapefile")
        start_time_cnt = time.time()
        county_shapefile_path = os.path.join(
            data_dir, "shp", "cb_2023_us_county_500k",
            "cb_2023_us_county_500k.shp"
        )
        counties_gdf = gpd.GeoDataFrame.from_file(county_shapefile_path)
        # Filter to Maine counties (STATEFP for Maine = 23)
        me_counties_gdf = counties_gdf.loc[
            counties_gdf["STATEFP"] == "23"
        ].copy()
        # Match CRS
        me_counties_gdf = me_counties_gdf.to_crs(me_gdf.crs)
        me_counties_gdf_str = me_counties_gdf.to_json()
        # Convert to Bokeh GeoJSON
        me_counties_src = GeoJSONDataSource(geojson=me_counties_gdf_str)

        elapsed_time_cnt = time.time() - start_time_cnt
        min = int(elapsed_time_cnt // 60)
        sec = np.round(elapsed_time_cnt % 60, 1)
        print(f"took {min} minutes and {sec} seconds.")

        # ---------------------------------------------------------------------
        # Create lake, reservoirs, and rivers shape file
        # ---------------------------------------------------------------------
        # Source: USGS National Hydrography Dataset (NHD) best resolution
        # Go to "https://www.usgs.gov/national-hydrography/" +
        # "access-national-hydrography-products", then click on "The National
        # Map Downloader" https://apps.nationalmap.gov/downloader/#/. Select
        # "Hydrography (3D Hydrography Program Products and Services)", then
        # check the boxes for "National Hydrography Dataset (NHD), "State", and
        # "Shapefile". Then click "Search Products". Download the zip file for
        # Maine.
        print("")
        print("Creating lakes and reservoirs shapefile")
        start_time_lk = time.time()
        lakes_res_riv_shapefile_path = os.path.join(
            data_dir, "shp",
            "NHD_H_Maine_State_Shape",
            "Shape",
            "NHDArea.shp"
        )
        lakes_res_riv_gdf = gpd.GeoDataFrame.from_file(
            lakes_res_riv_shapefile_path
        )

        # Ensure lakes, reservoirs, and rivers are in same CRS as Maine
        lakes_res_riv_gdf = lakes_res_riv_gdf.to_crs(me_gdf.crs)

        # Clip lakes/reservoirs to Maine boundary
        lakes_res_riv_me_gdf = gpd.clip(lakes_res_riv_gdf, me_gdf)

        # Find datetime-ish columns and convert to ISO strings
        dt_cols = lakes_res_riv_me_gdf.select_dtypes(
            include=["datetime64[ns]", "datetime64[ns, UTC]"]
        ).columns
        for c in dt_cols:
            lakes_res_riv_me_gdf[c] = lakes_res_riv_me_gdf[c].dt.strftime(
                "%Y-%m-%dT%H:%M:%S"
            )

        lakes_res_riv_me_gdf_str = lakes_res_riv_me_gdf.to_json()
        lakes_res_riv_me_src = GeoJSONDataSource(
            geojson=lakes_res_riv_me_gdf_str
        )

        elapsed_time_lk = time.time() - start_time_lk
        min = int(elapsed_time_lk // 60)
        sec = np.round(elapsed_time_lk % 60, 1)
        print(f"took {min} minutes and {sec} seconds.")

        # ---------------------------------------------------------------------
        # Save gdf and geojson data files
        # ---------------------------------------------------------------------
        # Create dictionaries of GeoDataFrames and GeoJSONDataSources for all
        # layers
        gdf_dict = {
            "me_gdf": me_gdf,
            "me_counties_gdf": me_counties_gdf,
            "lakes_res_riv_me_gdf": lakes_res_riv_me_gdf
        }
        geojson_dict = {
            "me_gdf_str": me_gdf_str,
            "me_counties_gdf_str": me_counties_gdf_str,
            "lakes_res_riv_me_gdf_str": lakes_res_riv_me_gdf_str
        }
        src_dict = {
            "me_src": me_src,
            "me_counties_src": me_counties_src,
            "lakes_res_riv_me_src": lakes_res_riv_me_src
        }
        if save_data:
            for name, gdf in gdf_dict.items():
                pickle.dump(
                    gdf, open(
                        os.path.join(data_dir, "gdf", f"{name}.pkl"), "wb"
                    )
                )
            for name, geojson in geojson_dict.items():
                with open(
                    os.path.join(data_dir, "geojson", f"{name}.geojson"),
                    "w", encoding="utf-8"
                ) as f:
                    f.write(geojson)

        elapsed_time_all = time.time() - start_time_all
        min = int(elapsed_time_all // 60)
        sec = np.round(elapsed_time_all % 60, 1)
        print("")
        print(f"Total data creation took {min} minutes and {sec} seconds.")
    else:
        print("")
        print("Reading in all the data from hard drive,")
        start_time = time.time()

        gdf_name_list = [
            "me_gdf", "me_counties_gdf", "lakes_res_riv_me_gdf"
        ]
        gdf_dict = {
            name: pickle.load(
                open(os.path.join(data_dir, "gdf", f"{name}.pkl"), "rb")
            ) for name in gdf_name_list
        }

        geojson_name_list = [
            "me_gdf_str", "me_counties_gdf_str", "lakes_res_riv_me_gdf_str"
        ]
        src_dict = {}
        for name in geojson_name_list:
            path = os.path.join(data_dir, "geojson", f"{name}.geojson")
            with open(path, "r", encoding="utf-8") as f:
                obj_str = f.read()
            obj_src = GeoJSONDataSource(geojson=obj_str)
            src_name = name.split("_gdf_str")[0] + "_src"
            src_dict[src_name] = obj_src

        elapsed_time = time.time() - start_time
        min = int(elapsed_time // 60)
        sec = np.round(elapsed_time % 60, 1)
        print(f"took {min} minutes and {sec} seconds.")

    # -------------------------------------------------------------------------
    # Merge property tax rates into counties GDF and assign Reds9 colors
    # Data are available at "https://www.maine.gov/revenue/sites/" +
    # "maine.gov.revenue/files/inline-files/" +
    # "2024%20FULL%20VALUE%20TAX%20RATES%20HISTORY%20with%20percentage%20increases.xls
    # -------------------------------------------------------------------------
    tax_df = pd.read_csv(
        os.path.join(data_dir, "ME_FullValuePropTaxRates_cnty_2024.csv"),
        skiprows=2
    ).dropna(subset=["county"])

    me_counties_gdf = gdf_dict["me_counties_gdf"].merge(
        tax_df[["county", "mill_rate_2024", "proptax_pct_2024"]],
        left_on="NAME",
        right_on="county",
        how="left"
    )

    # Reds9[8] = mill rate 7-7.99, Reds9[7] = 8-8.99, ..., Reds9[0] = 15-15.99
    def mill_rate_color(rate):
        if pd.isna(rate):
            return "gray"
        return Reds9[8 - int(np.clip(int(rate) - 7, 0, 8))]

    me_counties_gdf["color"] = me_counties_gdf["mill_rate_2024"].apply(
        mill_rate_color
    )

    # Rebuild counties source with tax data and color
    src_dict["me_counties_src"] = GeoJSONDataSource(
        geojson=me_counties_gdf.to_json()
    )

    # -------------------------------------------------------------------------
    # Make figure
    # -------------------------------------------------------------------------
    fig2_title = (
        "Figure 2. Maine map of 2024 full value property tax rates (mill " +
        "rates) by county"
    )

    # fig2_title = ""
    fig2_filename = "me_proptaxratemap.html"
    output_file(
        os.path.join(images_dir, fig2_filename),
        title=fig2_title, mode='inline'
    )

    TOOLS = "pan, box_zoom, wheel_zoom, save, reset, help"

    fig2 = figure(
        title=fig2_title,
        height=700,
        width=600,
        tools=TOOLS,
        min_border = 0,
        x_axis_location = None, y_axis_location = None,
        toolbar_location="right"
    )
    fig2.toolbar.logo = None
    fig2.grid.grid_line_color = None

    # Maine state outline
    print("Fig 2 Status: Plotting me_src")
    fig2.patches(
        "xs", "ys",
        source=src_dict["me_src"],
        fill_alpha=0.00,
        line_color="black",
        line_width=2,
        fill_color="white"
    )

    # Maine counties outline (colored by mill rate)
    print("Fig 2 Status: Plotting me_counties_src")
    r_counties = fig2.patches(
        "xs", "ys",
        source=src_dict["me_counties_src"],
        fill_color="color",
        fill_alpha=0.7,
        line_color="black",
        line_width=1
    )

    # Maine lakes / reservoirs
    print("Fig 2 Status: Plotting lakes_res_riv_me_src")
    fig2.patches(
        "xs", "ys",
        source=src_dict["lakes_res_riv_me_src"],
        fill_color="blue",
        fill_alpha=0.3,
        line_alpha=0.3,
        line_width=0.2
    )

    # Build 9-entry color legend for mill rate ranges
    legend_items = []
    for i in range(9):
        low = 7 + i
        dummy = fig2.rect(
            x=[float("nan")], y=[float("nan")], width=0, height=0,
            fill_color=Reds9[8 - i], fill_alpha=0.7,
            line_color="black", line_width=0.5
        )
        legend_items.append(
            LegendItem(label=f"{low}.00 to {low}.99", renderers=[dummy])
        )
    fig2_legend = Legend(
        items=legend_items,
        title="County mill rates"
    )
    fig2.add_layout(fig2_legend, 'right')

    # Legend properties
    fig2.legend.orientation = "vertical"
    fig2.legend.background_fill_color = "white"
    fig2.legend.background_fill_alpha = 0.9
    fig2.legend.border_line_color = "black"
    fig2.legend.border_line_width = 1
    fig2.legend.label_text_font_size = "10pt"
    fig2.legend.spacing = 2
    fig2.legend.padding = 6
    fig2.legend.margin = 6

    # Set up hover tool (counties only)
    hover2 = HoverTool(renderers=[r_counties])
    hover2.point_policy = "follow_mouse"
    hover2.tooltips = [
        ("County", "@NAME"),
        ("Mill Rate (2024)", "@mill_rate_2024{0.00}"),
        ("Property Tax Rate (2024)", "@proptax_pct_2024{0.000}%")
    ]
    fig2.add_tools(hover2)

    note_text_list2 = [
        (
            'Source: Richard W. Evans (@RickEcon), updated Apr. 20, 2026. ' +
            'Full value property tax rates are computed and'
        ),
        (
            '    published by the Maine Revenue Service and are available ' +
            'on the MRS website and in the data directory of'
        ),
        (
            '    the GitHub repository for this analysis.'
        )
    ]
    for note_text in note_text_list2:
        caption2 = Title(
            text=note_text, align='left', text_font_size='9pt',
            text_font_style='italic',
            text_color='black',
            standoff=0
        )
        fig2.add_layout(caption2, 'below')

    show(fig2)


if __name__ == "__main__":
    # make_me_elec_fmv_plot()
    make_me_datactrcostrevmap(create_data=True, save_data=True)
