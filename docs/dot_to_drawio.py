#!/usr/bin/env python3
"""Convert a dot-laid-out architecture graph to a draw.io XML file.

Pipeline:
    dot -Tjson architecture.dot -o architecture.json
    python dot_to_drawio.py architecture.json architecture.drawio

The graphviz JSON gives us the optimized layout (node positions and
spline control points for every edge). We re-emit those in mxGraph format
so draw.io renders the nodes natively (editable boxes) but routes the
edges along the graphviz-computed splines.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape


def parse_pos(pos: str) -> tuple[float, float]:
    x, y = pos.split(',')
    return float(x), float(y)


def parse_bb(bb: str) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in bb.split(','))
    return x1, y1, x2, y2


def cluster_style(stroke: str, fill: str) -> str:
    return (
        'rounded=1;whiteSpace=wrap;html=1;'
        f'fillColor={fill};strokeColor={stroke};'
        'verticalAlign=top;fontStyle=1;fontSize=12;'
    )


def node_style(stroke: str, fill: str) -> str:
    return (
        'rounded=1;whiteSpace=wrap;html=1;'
        f'fillColor={fill};strokeColor={stroke};'
        'fontSize=10;'
    )


def edge_style(color: str) -> str:
    # edgeStyle=none → use the explicit waypoints we provide.
    # curved=1       → smooth the polyline so spline curvature reads.
    # jumpStyle=arc  → arc line jumps over crossings.
    return (
        'edgeStyle=none;curved=1;html=1;'
        f'strokeColor={color};fontColor={color};'
        'fontSize=9;labelBackgroundColor=#FFFFFF;'
        'jumpStyle=arc;jumpSize=10;'
    )


# Match the colors used in architecture.dot's clusters / overrides.
CLUSTER_COLORS = {
    'cluster_arduino': ('#C68A1B', '#FFF4E5'),
    'cluster_vision':  ('#1F6FB2', '#E7F3FF'),
    'cluster_wskr':    ('#2D8A4E', '#E9F8EE'),
    'cluster_xarm':    ('#7A3FAE', '#F4ECFB'),
    'cluster_sysmgr':  ('#A07A00', '#FFF7E0'),
    'cluster_util':    ('#666666', '#F0F0F0'),
}

# Per-node fill overrides for the student-implemented stubs.
STUDENT_NODE_FILL = {'n_state', 'n_search'}


def main(in_path: str, out_path: str) -> None:
    with open(in_path, 'r', encoding='utf-8') as f:
        graph = json.load(f)

    _, _, canvas_w, canvas_h = parse_bb(graph['bb'])

    # graphviz Y is up, draw.io Y is down — flip about canvas_h.
    def flip(y: float) -> float:
        return canvas_h - y

    objects = graph.get('objects', [])
    edges = graph.get('edges', [])

    # Build a name → index lookup for edge endpoint resolution.
    idx_by_name = {o['name']: i for i, o in enumerate(objects)}

    cells: list[str] = [
        '<mxCell id="0" />',
        '<mxCell id="1" parent="0" />',
        # Legend
        (
            '<mxCell id="legend" '
            'value="&lt;b&gt;Edge legend&lt;/b&gt;&lt;br/&gt;'
            '&#160;&#160;&lt;span style=&#39;color:#000000&#39;&gt;━━&lt;/span&gt; topic&lt;br/&gt;'
            '&#160;&#160;&lt;span style=&#39;color:#C62828&#39;&gt;━━&lt;/span&gt; action&lt;br/&gt;'
            '&#160;&#160;&lt;span style=&#39;color:#1565C0&#39;&gt;━━&lt;/span&gt; service" '
            'style="text;html=1;align=left;verticalAlign=top;whiteSpace=wrap;fontSize=12;'
            'rounded=1;fillColor=#FFFFFF;strokeColor=#888888;" '
            'vertex="1" parent="1">'
            '<mxGeometry x="20" y="20" width="220" height="100" as="geometry" />'
            '</mxCell>'
        ),
    ]

    # ---------------- clusters ----------------
    for o in objects:
        name = o.get('name', '')
        if not name.startswith('cluster_'):
            continue
        bb = o.get('bb')
        if not bb:
            continue
        x1, y1, x2, y2 = parse_bb(bb)
        # graphviz cluster bb: top edge is y2, bottom edge is y1 (Y-up)
        x = x1
        y = flip(y2)
        w = x2 - x1
        h = y2 - y1

        stroke, fill = CLUSTER_COLORS.get(name, ('#888888', '#F8F8F8'))
        # Cluster label = the second-line italic in the dot file gets
        # collapsed; pull just the bold first line if present.
        label = o.get('label') or name.replace('cluster_', '')
        # Strip HTML wrappers; keep simple plain-text for label.
        for tag in ('<b>', '</b>', '<i>', '</i>', '<br/>'):
            label = label.replace(tag, ' ' if tag == '<br/>' else '')
        label = label.strip()

        style = cluster_style(stroke, fill)
        if name == 'cluster_sysmgr':
            style += 'dashed=1;'

        cells.append(
            f'<mxCell id="{escape(name)}" value="{escape(label)}" '
            f'style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x:.1f}" y="{y:.1f}" '
            f'width="{w:.1f}" height="{h:.1f}" as="geometry" />'
            '</mxCell>'
        )

    # ---------------- nodes -------------------
    for o in objects:
        name = o.get('name', '')
        if name.startswith('cluster_'):
            continue
        pos = o.get('pos')
        if not pos:
            continue
        cx, cy = parse_pos(pos)
        # graphviz width/height are in inches → convert to points (×72).
        w = float(o.get('width', 1.5)) * 72.0
        h = float(o.get('height', 0.6)) * 72.0
        # pos is the node center.
        x = cx - w / 2.0
        y = flip(cy) - h / 2.0

        # Pull the (HTML) label from the dot definition. Strip surrounding < >.
        label = (o.get('label') or name).strip()
        if label.startswith('<') and label.endswith('>'):
            label = label[1:-1]

        if name in STUDENT_NODE_FILL:
            fill, stroke = '#FFE9A8', '#A07A00'
        else:
            fill, stroke = '#F5F7FA', '#34495E'

        # Operator note keeps note shape.
        if name == 'n_extcmd':
            style = (
                'shape=note;whiteSpace=wrap;html=1;'
                'fillColor=#FAFAFA;strokeColor=#34495E;fontSize=11;'
            )
        else:
            style = node_style(stroke, fill)

        cells.append(
            f'<mxCell id="{escape(name)}" value="{escape(label)}" '
            f'style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x:.1f}" y="{y:.1f}" '
            f'width="{w:.1f}" height="{h:.1f}" as="geometry" />'
            '</mxCell>'
        )

    # ---------------- edges -------------------
    edge_counter = 0
    for e in edges:
        tail_idx = e['tail']
        head_idx = e['head']
        tail_name = objects[tail_idx]['name']
        head_name = objects[head_idx]['name']
        color = e.get('color', '#000000')
        label = (e.get('label') or '').replace('\\n', '\n')

        # Spline control points come out of _draw_ as a sequence of "b" ops
        # holding cubic-bezier control points. Concatenate them, drop the
        # tail/head endpoints (drawio already snaps to the source/target),
        # and use the interior points as waypoints.
        spline_pts: list[tuple[float, float]] = []
        for cmd in e.get('_draw_', []):
            if cmd.get('op') in ('b', 'B'):
                for px, py in cmd.get('points', []):
                    spline_pts.append((px, flip(py)))
        # Endpoints land on the node bounding box. Skip first and last so
        # drawio's port-snap handles attachment, and use the interior
        # control points as waypoints.
        waypoints = spline_pts[1:-1] if len(spline_pts) >= 3 else []

        edge_id = f'e{edge_counter:03d}'
        edge_counter += 1

        # Place the label at graphviz's chosen label point, if available.
        lp = e.get('lp')
        if lp and label:
            lpx, lpy = parse_pos(lp)
            lpx_d, lpy_d = lpx, flip(lpy)
        else:
            lpx_d = lpy_d = None

        wp_xml = ''
        if waypoints:
            inner = ''.join(
                f'<mxPoint x="{x:.1f}" y="{y:.1f}" />' for x, y in waypoints
            )
            wp_xml = f'<Array as="points">{inner}</Array>'

        cells.append(
            f'<mxCell id="{edge_id}" value="{escape(label)}" '
            f'style="{edge_style(color)}" '
            f'edge="1" parent="1" '
            f'source="{escape(tail_name)}" target="{escape(head_name)}">'
            f'<mxGeometry relative="1" as="geometry">'
            f'{wp_xml}'
            f'</mxGeometry>'
            '</mxCell>'
        )

    inner_xml = '\n        '.join(cells)
    page_w = int(canvas_w + 80)
    page_h = int(canvas_h + 80)

    drawio = f'''<mxfile host="app.diagrams.net" agent="dot_to_drawio" version="22.1.0" type="device">
  <diagram name="Project 6 architecture" id="proj6-arch">
    <mxGraphModel dx="2400" dy="1600" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{page_w}" pageHeight="{page_h}" math="0" shadow="0">
      <root>
        {inner_xml}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
'''

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(drawio)
    print(f'Wrote {out_path}: {len(objects)} objects, {len(edges)} edges.')


if __name__ == '__main__':
    in_path = sys.argv[1] if len(sys.argv) > 1 else 'architecture.json'
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'architecture.drawio'
    main(in_path, out_path)
