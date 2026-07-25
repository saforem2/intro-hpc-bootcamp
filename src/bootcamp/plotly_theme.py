"""
bootcamp/plotly_theme.py

A shared Plotly theme matching the site's `ambivalent` matplotlib style:
transparent backgrounds, #838383 grey text/grid, the material color cycle, and
the "Iosevka Web" font (loaded site-wide via webfont link).

Usage (top of any notebook page):

    from bootcamp.plotly_theme import apply_theme
    apply_theme()                      # registers + activates the "ambivalent" template
    import plotly.graph_objects as go
    go.Figure(...)                     # inherits the theme automatically

Every figure then shares one look, so the rollout is a one-line import per page.
"""
from __future__ import annotations

# The material palette used by the `ambivalent` matplotlib style (in cycle order).
COLORWAY = [
    "#2196F3",  # blue
    "#EF5350",  # red
    "#4CAF50",  # green
    "#FFA726",  # orange
    "#AE81FF",  # purple
    "#ffeb3b",  # yellow
    "#EC407A",  # pink
    "#009688",  # teal
    "#838383",  # grey
]

GREY = "#838383"                 # text / axis / grid color
FONT = '"Iosevka Web", ui-monospace, "Cascadia Code", monospace'
TRANSPARENT = "rgba(0,0,0,0)"    # transparent bg (light + dark friendly)

# Semantic aliases so pages can pull individual house colors by name.
COLORS = {
    "blue": "#2196F3", "red": "#EF5350", "green": "#4CAF50",
    "orange": "#FFA726", "purple": "#AE81FF", "yellow": "#ffeb3b",
    "pink": "#EC407A", "teal": "#009688", "grey": GREY,
}


def ambivalent_template():
    """Return a plotly.graph_objects.layout.Template for the house style."""
    import plotly.graph_objects as go

    axis = dict(
        gridcolor="rgba(131,131,131,0.2)",   # #838383 @ 20%
        zerolinecolor="rgba(131,131,131,0.3)",
        linecolor="rgba(131,131,131,0.4)",
        tickcolor="rgba(131,131,131,0.4)",
        title_font=dict(color=GREY),
        tickfont=dict(color=GREY),
    )
    return go.layout.Template(
        layout=dict(
            paper_bgcolor=TRANSPARENT,
            plot_bgcolor=TRANSPARENT,
            font=dict(family=FONT, color=GREY, size=13),
            title_font=dict(family=FONT, color=GREY),
            colorway=COLORWAY,
            xaxis=axis,
            yaxis=axis,
            legend=dict(font=dict(color=GREY), bgcolor=TRANSPARENT),
            margin=dict(l=0, r=0, t=30, b=0),
            colorscale=dict(sequential="Viridis"),
            hoverlabel=dict(font=dict(family=FONT)),
        )
    )


def apply_theme(name: str = "ambivalent", default: bool = True, embed: bool = True):
    """Register the house template with Plotly and make it the default.

    Also sets the renderer so Quarto **inlines the full plotly.js** into each page
    (`...+notebook`) instead of loading it from cdn.plot.ly (`...+notebook_connected`).
    That makes the built site self-contained: charts work offline and don't break
    if the CDN is down/blocked. Pass embed=False to fall back to the CDN.

    Returns the template name so callers can pass `template=name` explicitly if
    they prefer not to rely on the global default.
    """
    import plotly.io as pio

    pio.templates[name] = ambivalent_template()
    if default:
        # compose over plotly_white so we keep sensible defaults we didn't set
        pio.templates.default = f"plotly_white+{name}"
    if embed:
        # "notebook" inlines the library; "notebook_connected" would use the CDN
        pio.renderers.default = "plotly_mimetype+notebook"
    return name
