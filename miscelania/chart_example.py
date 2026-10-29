import sys
from html import escape
from pathlib import Path

from graphviz import Digraph, ExecutableNotFound

OUT_DIR = Path(__file__).resolve().parent
OUT_NAME = "un_org_chart"          # -> un_org_chart.png
DPI = "200"
SHOW_LEGEND = True
LAYOUT = "vertical"                # "vertical" (tall, fits A4 portrait) or "horizontal" (wide)
ORGAN_BOX_WIDTH = 2.6              # inches; all organ boxes get (at least) this width
BODY_BOX_WIDTH = 2.7               # inches; same for the agencies column
if len(sys.argv) > 1 and sys.argv[1] in ("vertical", "horizontal"):
    LAYOUT = sys.argv[1]

# ----------------------------------------------------------------------------
# DATA  (id, display name, years of activity)
# Years for the organs are the years they started to work (first sessions, 1946/47),
# the UN Charter itself entered into force on 24 October 1945.
# ----------------------------------------------------------------------------
ORGANS = [
    ("ga",     "General Assembly",                  "1946 – present"),
    ("sc",     "Security Council",                  "1946 – present"),
    ("ecosoc", "Economic and Social\nCouncil (ECOSOC)", "1946 – present"),
    ("tc",     "Trusteeship Council",               "1947 – 1994 (suspended)"),
    ("icj",    "International Court\nof Justice (ICJ)", "1946 – present"),
    ("sec",    "Secretariat",                       "1946 – present"),
]

# Bodies attached to an organ: (id, name, years, kind, parent id)
# kind decides colour and line style: programme | agency | office | service
BODIES = [
    ("unep",   "UN Environment\nProgramme (UNEP)",
     "1972 – present", "programme", "ga"),
    ("unesco", "UNESCO",
     "1945 – present", "agency", "ecosoc"),
    ("ilo",    "International Labour\nOrganization (ILO)",
     "1919 – present\n(UN agency since 1946)", "agency", "ecosoc"),
    ("ohchr",  "Office of the UN High\nCommissioner for\nHuman Rights (OHCHR)",
     "1993 – present", "office", "sec"),
    ("treaty", "UN Treaty Collection\n(Treaty Section,\nOffice of Legal Affairs)",
     "UN Treaty Series:\n1946 – present", "service", "sec"),
]

# ----------------------------------------------------------------------------
# STYLE
# ----------------------------------------------------------------------------
UN_BLUE = "#009EDB"
KIND_STYLE = {
    "root":      dict(fill=UN_BLUE,   font="white",   border="#006C94", line="solid"),
    "organ":     dict(fill="#CFEAF7", font="#0B2E4A", border="#009EDB", line="solid"),
    "ended":     dict(fill="#E6E6E6", font="#555555", border="#999999", line="dashed"),
    "programme": dict(fill="#FFF1C2", font="#4A3B00", border="#D4A800", line="solid"),
    "agency":    dict(fill="#D9F0D3", font="#1C4A12", border="#4C9A3C", line="dashed"),
    "office":    dict(fill="#FFE0CC", font="#5A2A0A", border="#E07A33", line="solid"),
    "service":   dict(fill="#EADCF5", font="#3B1E55", border="#8A5CB8", line="dotted"),
}
EDGE_STYLE = {            # style of the line coming from the parent
    "programme": "solid",
    "agency":    "dashed",   # autonomous bodies, linked by agreement
    "office":    "solid",
    "service":   "solid",
}


def html_label(name: str, years: str | None = None) -> str:
    """Bold name + smaller years, as a Graphviz HTML-like label."""
    name_html = "<BR/>".join(escape(part) for part in name.split("\n"))
    out = f"<<B>{name_html}</B>"
    if years:
        years_html = "<BR/>".join(escape(part) for part in years.split("\n"))
        out += f'<BR/><FONT POINT-SIZE="10">{years_html}</FONT>'
    return out + ">"


def add_node(g, node_id: str, name: str, years: str | None, kind: str,
             min_width: float | None = None):
    s = KIND_STYLE[kind]
    extra = {"width": str(min_width)} if min_width else {}
    g.node(
        node_id,
        label=html_label(name, years),
        shape="box",
        style=f"rounded,filled,{s['line']}",
        fillcolor=s["fill"],
        fontcolor=s["font"],
        color=s["border"],
        penwidth="1.6",
        margin="0.25,0.14",
        **extra,
    )


def legend_label(columns: int) -> str:
    cells = [
        ("Principal organ", "organ"),
        ("Ceased / suspended organ", "ended"),
        ("Programme (GA subsidiary)", "programme"),
        ("Specialized agency", "agency"),
        ("Secretariat office", "office"),
        ("Service / database", "service"),
    ]
    tds = [
        f'<TD BGCOLOR="{KIND_STYLE[k]["fill"]}" BORDER="1" COLOR="{KIND_STYLE[k]["border"]}" '
        f'CELLPADDING="5" ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="{KIND_STYLE[k]["font"]}">'
        f'{escape(t)}</FONT></TD>'
        for t, k in cells
    ]
    rows = "".join(
        "<TR>" + "".join(tds[i:i + columns]) + "</TR>" for i in range(0, len(tds), columns)
    )
    return f'<<TABLE BORDER="0" CELLSPACING="6">{rows}</TABLE>>'


def bus(g, group, parent: str, children: list[tuple[str, str]], tag: str, penwidth: str):
    """
    Trunk-and-branch connector: ONE line leaves `parent`, runs into a bar, and every
    child hangs from that bar. `children` = [(child_id, line_style), ...].
    Invisible point nodes (the corners of the bar) live in `group`, a same-rank
    subgraph, so the bar is straight and every branch is a straight line.
    """
    n = len(children)
    js = [f"j_{tag}_{cid}" for cid, _ in children]
    chain = list(js)
    if n % 2 == 1:
        entry = js[n // 2]                      # the middle child's own corner
    else:
        entry = f"j_{tag}_mid"                  # extra point between the two middle ones
        chain.insert(n // 2, entry)
    dot = f"{float(penwidth) * 1.15 / 72:.4f}"          # same thickness as the line (1 pt = 1/72 in)
    for j in chain:                             # tiny dots in the line colour seal the joints
        group.node(j, shape="point", width=dot, height=dot, color="#555555", penwidth="0")

    styles = {st for _, st in children}
    trunk_style = styles.pop() if len(styles) == 1 else "solid"

    for a, b in zip(chain, chain[1:]):          # the bar
        g.edge(a, b, style=trunk_style, penwidth=penwidth, weight="1")
    g.edge(parent, entry, style=trunk_style, penwidth=penwidth, weight="10")   # the stem
    for j, (cid, st) in zip(js, children):      # the branches
        g.edge(j, cid, style=st, penwidth=penwidth, weight="100")
    return chain


def build() -> Digraph:
    vertical = LAYOUT == "vertical"

    g = Digraph("UN", format="png")
    g.attr(
        # vertical page = root on the left, organs stacked in a column, bodies on the right
        rankdir="LR" if vertical else "TB",
        splines="line",                       # all lines are straight (horizontal / vertical)
        nodesep="0.3" if vertical else "0.35",
        ranksep="0.35" if vertical else "0.45",
        dpi=DPI,
        bgcolor="white",
        pad="0.3",
        fontname="Helvetica",
    )
    g.attr("node", fontname="Helvetica", fontsize="12")
    g.attr("edge", color="#555555", arrowhead="none", penwidth="1.4")

    if SHOW_LEGEND:
        g.attr(label=legend_label(columns=2 if vertical else 6),
               labelloc="b", labeljust="c")

    organ_w = ORGAN_BOX_WIDTH if vertical else None      # equal widths -> aligned columns
    body_w = BODY_BOX_WIDTH if vertical else None

    # one same-rank group per "column" (vertical layout) / "row" (horizontal layout)
    with g.subgraph() as organs, g.subgraph() as bodies, \
            g.subgraph() as bus_organs, g.subgraph() as bus_bodies:
        for grp in (organs, bodies, bus_organs, bus_bodies):
            grp.attr(rank="same")

        add_node(g, "un", "United Nations", "Charter in force\n24 Oct 1945 – present", "root")
        for oid, name, years in ORGANS:
            add_node(organs, oid, name, years, "ended" if oid == "tc" else "organ", organ_w)
        for bid, name, years, kind, _parent in BODIES:
            add_node(bodies, bid, name, years, kind, body_w)

        # UN -> all six organs: a single stem that splits into a bar
        bus(g, bus_organs, "un",
            [(oid, "dashed" if oid == "tc" else "solid") for oid, *_ in ORGANS],
            tag="un", penwidth="2.0")

        # organ -> its agencies / programmes / offices / services
        chains = []
        for oid, *_ in ORGANS:
            kids = [(bid, EDGE_STYLE[kind]) for bid, _n, _y, kind, parent in BODIES
                    if parent == oid]
            if kids:
                chains.append(bus(g, bus_bodies, oid, kids, tag=oid, penwidth="1.4"))

        # keep the agencies in the order of BODIES (= order of their parent organs);
        # invisible edges only fix the order, nothing is drawn
        for (a, *_), (b, *_) in zip(BODIES, BODIES[1:]):
            g.edge(a, b, style="invis", weight="1")
        for prev, nxt in zip(chains, chains[1:]):
            g.edge(prev[-1], nxt[0], style="invis", weight="1")

    return g


def main() -> None:
    graph = build()
    name = OUT_NAME if LAYOUT == "vertical" else f"{OUT_NAME}_horizontal"
    try:
        path = graph.render(filename=name, directory=str(OUT_DIR), format="png", cleanup=True)
    except ExecutableNotFound:
        raise SystemExit(
            "Graphviz 'dot' program not found. Install Graphviz and make sure it is on "
            "your PATH (see the instructions at the top of this file), then run again."
        )
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()