import numpy as np
import panel as pn

pn.extension(sizing_mode="stretch_width")


def elec_fmv_proptax_func(
    elec_capac: int | float = 100,
    tax_exempt_pct: int | float = 0.0,
    county: str = "Cumberland",
):
    a_lin = 0.010018082083149032
    b_lin = -0.3496020387036265
    a_exp = 0.010396604326045984
    b_exp = -1.0247649882273389
    c_exp = -0.36147598374031675
    ElecCapacCutoff = 95
    proptax_rate_2024_dict = {
        "Androscoggin": 0.01285, "Aroostook": 0.01596, "Cumberland": 0.01062,
        "Franklin": 0.00859, "Hancock": 0.00839, "Kennebec": 0.01107,
        "Knox": 0.01058, "Lincoln": 0.00753, "Oxford": 0.0094,
        "Penobscot": 0.0138, "Piscataquis": 0.01004, "Sagadahoc": 0.01035,
        "Somerset": 0.01185, "Waldo": 0.01137, "Washington": 0.01188,
        "York": 0.00876,
    }
    if elec_capac >= ElecCapacCutoff:
        fmv = a_lin * elec_capac + b_lin
    else:
        fmv = np.exp(a_exp * elec_capac + b_exp) + c_exp
    proptax_rate = proptax_rate_2024_dict[county]
    proptax_revenue = fmv * proptax_rate * (1 - tax_exempt_pct)
    return float(fmv), proptax_rate, float(proptax_revenue)

MAINE_COUNTIES = [
    "Androscoggin", "Aroostook", "Cumberland", "Franklin", "Hancock",
    "Kennebec", "Knox", "Lincoln", "Oxford", "Penobscot", "Piscataquis",
    "Sagadahoc", "Somerset", "Waldo", "Washington", "York",
]

# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------
elec_slider = pn.widgets.FloatSlider(
    name="Data Center Electrical Capacity (megawatts, MW)",
    start=0.9, end=2200.0, step=0.1, value=10.0,
    bar_color="#0D9488",
)

county_select = pn.widgets.Select(
    name="Data Center Location, Maine County",
    options=MAINE_COUNTIES,
    value="Cumberland",
)

tax_slider = pn.widgets.FloatSlider(
    name="Tax Exempt Percent  (0.00 – 1.00)",
    start=0.0, end=1.0, step=0.01, value=0.0,
    bar_color="#0D9488",
    format="0.00",
)

# ---------------------------------------------------------------------------
# Reactive output cards
# ---------------------------------------------------------------------------
@pn.depends(elec_slider, county_select, tax_slider)
def results_pane(elec_capac, county, tax_exempt_pct):
    fmv, proptax_rate, proptax_revenue = elec_fmv_proptax_func(
        elec_capac=elec_capac,
        tax_exempt_pct=tax_exempt_pct,
        county=county,
    )
    html = f"""
    <div style="display:flex; flex-direction:column; gap:18px; padding:8px 0;">

      <div style="background:#0D9488; color:#FFFFFF; border-radius:14px;
                  padding:26px 28px; text-align:center;
                  box-shadow:0 4px 14px rgba(13,148,136,0.35);">
        <div style="font-size:12px; font-weight:700; text-transform:uppercase;
                    letter-spacing:1.4px; opacity:0.88; margin-bottom:10px;">
          Data Center Fair Market Value (FMV)
        </div>
        <div style="font-size:38px; font-weight:800; letter-spacing:0.5px;">
          ${fmv * 1e9:,.0f}
        </div>
      </div>

      <div style="background:#1B3A5C; color:#FFFFFF; border-radius:14px;
                  padding:26px 28px; text-align:center;
                  box-shadow:0 4px 14px rgba(27,58,92,0.30);">
        <div style="font-size:12px; font-weight:700; text-transform:uppercase;
                    letter-spacing:1.4px; opacity:0.88; margin-bottom:10px;">
          {county} County Average Property Tax Rate
        </div>
        <div style="font-size:38px; font-weight:800;">
          {proptax_rate:.2%}
        </div>
      </div>

      <div style="background:#F59E0B; color:#1B3A5C; border-radius:14px;
                  padding:26px 28px; text-align:center;
                  box-shadow:0 4px 14px rgba(245,158,11,0.35);">
        <div style="font-size:12px; font-weight:700; text-transform:uppercase;
                    letter-spacing:1.4px; opacity:0.85; margin-bottom:10px;">
          Projected Annual County Property Tax Revenue from Data Center
        </div>
        <div style="font-size:38px; font-weight:800;">
          ${proptax_revenue * 1e9:,.0f}
        </div>
      </div>

    </div>
    """
    return pn.pane.HTML(html, sizing_mode="stretch_width")

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
CARD_STYLE = {
    "background": "#FFFFFF",
    "border-radius": "14px",
    "padding": "28px 32px",
    "box-shadow": "0 4px 16px rgba(0,0,0,0.07)",
}

input_section_header = pn.pane.HTML(
    '<p style="color:#1B3A5C; font-size:13px; font-weight:700; '
    'text-transform:uppercase; letter-spacing:1.4px; margin:0 0 18px 0;">'
    'Inputs</p>'
)

input_panel = pn.Column(
    input_section_header,
    elec_slider,
    pn.Spacer(height=10),
    county_select,
    pn.Spacer(height=10),
    tax_slider,
    styles=CARD_STYLE,
    sizing_mode="stretch_width",
)

arrow_pane = pn.pane.HTML(
    '<div style="display:flex; align-items:center; justify-content:center; '
    'height:100%; font-size:52px; color:#0D9488; padding:0 8px;">&#10140;</div>',
    width=70,
    sizing_mode="fixed",
)

output_section_header = pn.pane.HTML(
    '<p style="color:#1B3A5C; font-size:13px; font-weight:700; '
    'text-transform:uppercase; letter-spacing:1.4px; margin:0 0 8px 0;">'
    'Outputs</p>'
)

output_panel = pn.Column(
    output_section_header,
    results_pane,
    styles=CARD_STYLE,
    sizing_mode="stretch_width",
)

header = pn.pane.HTML("""
<div style="background:linear-gradient(135deg, #1B3A5C 0%, #0D9488 100%);
            padding:28px 40px; color:white; border-radius:0 0 0 0;
            margin-bottom:28px;">
  <h1 style="margin:0 0 6px 0; font-size:26px; font-weight:800; letter-spacing:0.3px;">
    Maine Data Center Property Tax Calculator
  </h1>
  <p style="margin:0; font-size:14px; opacity:0.82;">
    Estimate fair market value and annual property tax revenue for a Maine data center
  </p>
</div>
""")

footer = pn.pane.HTML("""
<div style="margin-top:28px; padding:16px 0; border-top:1px solid #CBD5E1;
            font-size:11px; color:#64748B; text-align:center;">
  Richard W. Evans (@RickEcon) &nbsp;|&nbsp;
  Source: <a href="https://github.com/OpenSourceEcon/ME-DataCtrCostRev"
             style="color:#0D9488; text-decoration:none;">
    OpenSourceEcon/ME-DataCtrCostRev
  </a>
</div>
""")

body = pn.Row(
    input_panel,
    arrow_pane,
    output_panel,
    sizing_mode="stretch_width",
)

app = pn.Column(
    header,
    body,
    footer,
    styles={"background": "#F0F4F8", "padding": "0 32px 32px"},
    sizing_mode="stretch_width",
)

app.servable()

if __name__ == "__main__":
    pn.serve(app, port=5006, show=True)
