"""
drawio2pptx.py  —  Convert a .drawio file (multi-page) to a .pptx file.
Each diagram page becomes one slide.
Handles: rectangles, rounded rects, diamonds, ellipses, swimlanes,
         parallelograms, parent-relative coords, edge connectors, HTML text.
"""

import re, html as html_mod, xml.etree.ElementTree as ET
import os
from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from lxml import etree

# ── PPTX slide size (landscape A4: 33.87 cm × 23.99 cm) ──────────────────────
SLIDE_W_EMU = int(33.87 / 2.54 * 914400)   # ≈ 12192000
SLIDE_H_EMU = int(23.99 / 2.54 * 914400)   # ≈  8636400

MARGIN = int(0.5 / 2.54 * 914400)          # 5 mm margin


# ── Style parsing ─────────────────────────────────────────────────────────────
def parse_style(s):
    d = {}
    for part in (s or '').split(';'):
        part = part.strip()
        if not part:
            continue
        if '=' in part:
            k, v = part.split('=', 1)
            d[k.strip()] = v.strip()
        else:
            d[part] = 'true'
    return d

def hex_color(s, default=None):
    if not s or not s.startswith('#'):
        return default
    h = s.lstrip('#')
    if len(h) == 6:
        return RGBColor(int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))
    return default

def text_from_value(val):
    """Strip HTML tags, decode entities, collapse whitespace."""
    if not val:
        return ''
    # Remove HTML tags
    text = re.sub(r'<br\s*/?>', '\n', val, flags=re.IGNORECASE)
    text = re.sub(r'<div>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = html_mod.unescape(text)
    # Collapse non-newline whitespace
    lines = [re.sub(r'[ \t]+', ' ', ln).strip() for ln in text.splitlines()]
    return '\n'.join(l for l in lines if l)

def get_shape_type(sd):
    """Map draw.io style dict to a prstGeom name."""
    if 'rhombus' in sd:
        return 'diamond'
    if 'ellipse' in sd:
        return 'ellipse'
    shape = sd.get('shape', '')
    if 'parallelogram' in shape:
        return 'parallelogram'
    if 'cylinder' in shape:
        return 'cylinder'
    if 'triangle' in shape:
        return 'rightTriangle'
    if 'component' in shape:
        return 'rect'
    if 'note' in shape:
        return 'note'
    if sd.get('rounded') == '1':
        return 'roundRect'
    return 'rect'


# ── PPTX shape builder ────────────────────────────────────────────────────────
def set_prst_geom(shape, prst):
    sp = shape.element
    prstGeom = sp.find('.//' + qn('a:prstGeom'))
    if prstGeom is not None:
        prstGeom.set('prst', prst)
        # Clear any existing avLst
        for child in list(prstGeom):
            prstGeom.remove(child)
        avLst = etree.SubElement(prstGeom, qn('a:avLst'))
        if prst == 'roundRect':
            gd = etree.SubElement(avLst, qn('a:gd'))
            gd.set('name', 'adj')
            gd.set('fmla', 'val 25000')

def make_shape(slide, prst, x, y, w, h, fill_rgb, stroke_rgb, opacity=100, dashed=False):
    shape = slide.shapes.add_shape(1, x, y, w, h)
    set_prst_geom(shape, prst)

    # Fill
    if fill_rgb:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_rgb
    else:
        shape.fill.background()

    # Opacity via XML
    if opacity < 100 and fill_rgb:
        sp = shape.element
        solidFill = sp.find('.//' + qn('a:solidFill'))
        if solidFill is not None:
            srgbClr = solidFill.find(qn('a:srgbClr'))
            if srgbClr is not None:
                alpha = etree.SubElement(srgbClr, qn('a:alpha'))
                alpha.set('val', str(int(opacity * 1000)))

    # Border
    if stroke_rgb:
        shape.line.color.rgb = stroke_rgb
    shape.line.width = Pt(1)
    if dashed:
        from pptx.oxml.ns import nsmap
        sp = shape.element
        ln = sp.find('.//' + qn('a:ln'))
        if ln is not None:
            prstDash = etree.SubElement(ln, qn('a:prstDash'))
            prstDash.set('val', 'dash')

    return shape

def set_text(shape, text, font_size=10, bold=False, font_color=None, align=PP_ALIGN.CENTER):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_top = tf.margin_bottom = Pt(3)
    tf.margin_left = tf.margin_right = Pt(4)

    lines = text.split('\n') if text else ['']
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.size = Pt(max(font_size, 6))
        run.font.bold = bold
        if font_color:
            run.font.color.rgb = font_color


# ── Orthogonal edge routing ───────────────────────────────────────────────────

def _side_from_xy(ex, ey):
    """Convert fractional exit/entry coords to a side name."""
    ex, ey = float(ex), float(ey)
    if ey == 0.0:   return 'top'
    if ey == 1.0:   return 'bottom'
    if ex == 0.0:   return 'left'
    if ex == 1.0:   return 'right'
    # intermediate values → pick dominant axis
    return 'bottom' if ey >= 0.5 else 'top'

def _border_pt(ax, ay, aw, ah, side, to_emu_x, to_emu_y):
    """EMU coords of the connection point on the given side."""
    if side == 'bottom': return to_emu_x(ax + aw / 2), to_emu_y(ay + ah)
    if side == 'top':    return to_emu_x(ax + aw / 2), to_emu_y(ay)
    if side == 'right':  return to_emu_x(ax + aw),     to_emu_y(ay + ah / 2)
    if side == 'left':   return to_emu_x(ax),           to_emu_y(ay + ah / 2)
    return to_emu_x(ax + aw / 2), to_emu_y(ay + ah / 2)

def _route(x1, y1, exit_side, x2, y2, entry_side):
    """
    Build an orthogonal polyline (list of EMU points) from (x1,y1) to (x2,y2).
    Every segment will be either perfectly horizontal or perfectly vertical.
    """
    TOL = 40000   # ~4 mm: treat as "aligned" if within this
    pts = [(x1, y1)]

    es, en = exit_side, entry_side

    # ── Same-direction straight cases ────────────────────────────────────────
    if es == 'bottom' and en == 'top':
        if abs(x1 - x2) < TOL:          # already aligned
            pts.append((x1, y2))
        else:
            mid = (y1 + y2) // 2
            pts += [(x1, mid), (x2, mid), (x2, y2)]

    elif es == 'top' and en == 'bottom':
        if abs(x1 - x2) < TOL:
            pts.append((x1, y2))
        else:
            mid = (y1 + y2) // 2
            pts += [(x1, mid), (x2, mid), (x2, y2)]

    elif es == 'right' and en == 'left':
        if abs(y1 - y2) < TOL:
            pts.append((x2, y1))
        else:
            mid = (x1 + x2) // 2
            pts += [(mid, y1), (mid, y2), (x2, y2)]

    elif es == 'left' and en == 'right':
        if abs(y1 - y2) < TOL:
            pts.append((x2, y1))
        else:
            mid = (x1 + x2) // 2
            pts += [(mid, y1), (mid, y2), (x2, y2)]

    # ── Corner cases (L-turn) ─────────────────────────────────────────────────
    elif es == 'bottom' and en == 'left':
        pts += [(x1, y2), (x2, y2)]
    elif es == 'bottom' and en == 'right':
        pts += [(x1, y2), (x2, y2)]
    elif es == 'top' and en == 'left':
        pts += [(x1, y2), (x2, y2)]
    elif es == 'top' and en == 'right':
        pts += [(x1, y2), (x2, y2)]
    elif es == 'right' and en == 'top':
        pts += [(x2, y1), (x2, y2)]
    elif es == 'right' and en == 'bottom':
        pts += [(x2, y1), (x2, y2)]
    elif es == 'left' and en == 'top':
        pts += [(x2, y1), (x2, y2)]
    elif es == 'left' and en == 'bottom':
        pts += [(x2, y1), (x2, y2)]

    else:
        # Fallback: go to same y-level then horizontal
        pts += [(x1, y2), (x2, y2)]

    return pts

def _fix_exit(pts, x1, y1, exit_side, GAP=200000):
    """Ensure the first segment departs in the correct exit direction."""
    if len(pts) < 2:
        return pts
    nx, ny = pts[1]
    ok = (
        (exit_side == 'bottom' and ny > y1) or
        (exit_side == 'top'    and ny < y1) or
        (exit_side == 'right'  and nx > x1) or
        (exit_side == 'left'   and nx < x1)
    )
    if ok:
        return pts
    # Build a stub in the correct exit direction, then turn toward pts[1]
    if exit_side == 'bottom':   stub = (x1, y1 + GAP)
    elif exit_side == 'top':    stub = (x1, y1 - GAP)
    elif exit_side == 'right':  stub = (x1 + GAP, y1)
    else:                       stub = (x1 - GAP, y1)
    sx, sy = stub
    if sx != nx and sy != ny:
        corner = (nx, sy) if exit_side in ('bottom', 'top') else (sx, ny)
        return [pts[0], stub, corner] + list(pts[1:])
    return [pts[0], stub] + list(pts[1:])


def _fix_approach(pts, x2, y2, entry_side, GAP=200000):
    """Ensure the last segment approaches the target from the correct direction."""
    if len(pts) < 2:
        return pts
    prev_x, prev_y = pts[-2]
    ok = (
        (entry_side == 'top'    and prev_y < y2) or
        (entry_side == 'bottom' and prev_y > y2) or
        (entry_side == 'right'  and prev_x > x2) or
        (entry_side == 'left'   and prev_x < x2)
    )
    if ok:
        return pts
    # Insert a stub just before (x2,y2) from the correct approach direction
    if entry_side == 'top':      stub = (x2, y2 - GAP)
    elif entry_side == 'bottom': stub = (x2, y2 + GAP)
    elif entry_side == 'right':  stub = (x2 + GAP, y2)
    else:                        stub = (x2 - GAP, y2)
    sx, sy = stub
    if prev_x != sx and prev_y != sy:
        corner = (prev_x, sy) if entry_side in ('top', 'bottom') else (sx, prev_y)
        return list(pts[:-1]) + [corner, stub, (x2, y2)]
    return list(pts[:-1]) + [stub, (x2, y2)]


def _draw_polyline(slide, pts, stroke_rgb, dashed, width_pt):
    """
    Draw an orthogonal polyline as multiple straight connector segments.
    Only the final segment gets an arrowhead.
    """
    for i in range(len(pts) - 1):
        x1, y1 = int(pts[i][0]),   int(pts[i][1])
        x2, y2 = int(pts[i+1][0]), int(pts[i+1][1])
        # Avoid zero-length segments
        if x1 == x2 and y1 == y2:
            continue
        try:
            conn = slide.shapes.add_connector(1, x1, y1, x2, y2)
            if stroke_rgb:
                conn.line.color.rgb = stroke_rgb
            conn.line.width = Pt(width_pt)

            sp  = conn.element
            ln  = sp.find('.//' + qn('a:ln'))
            if ln is None:
                continue

            if dashed:
                pd = etree.SubElement(ln, qn('a:prstDash'))
                pd.set('val', 'dash')

            is_last = (i == len(pts) - 2)
            if is_last:
                # arrowhead at the end of the last segment (tailEnd = end/x2,y2 side)
                head = etree.SubElement(ln, qn('a:tailEnd'))
                head.set('type', 'arrow')
                head.set('w', 'med')
                head.set('len', 'med')
        except Exception:
            pass


# ── Coordinate helpers ────────────────────────────────────────────────────────
def resolve_geometry(cell_id, cells_by_id):
    """Return absolute (x, y, w, h) in draw.io units for a cell."""
    cell = cells_by_id.get(cell_id)
    if cell is None:
        return 0, 0, 0, 0
    geo = cell.find('mxGeometry')
    if geo is None:
        return 0, 0, 0, 0
    x = float(geo.get('x') or 0)
    y = float(geo.get('y') or 0)
    w = float(geo.get('width') or 0)
    h = float(geo.get('height') or 0)

    parent_id = cell.get('parent', '1')
    if parent_id not in ('0', '1'):
        px, py, _, _ = resolve_geometry(parent_id, cells_by_id)
        x += px
        y += py
    return x, y, w, h


def scale(val, factor):
    return int(val * factor)


# ── Main diagram → slide converter ────────────────────────────────────────────
def diagram_to_slide(diagram_elem, prs):
    mg = diagram_elem.find('mxGraphModel')
    if mg is None:
        return

    page_w = float(mg.get('pageWidth', 827))
    page_h = float(mg.get('pageHeight', 1169))

    slide_layout = prs.slide_layouts[6]   # blank
    slide = prs.slides.add_slide(slide_layout)

    root_elem = mg.find('root')
    if root_elem is None:
        return

    all_cells = root_elem.findall('mxCell')
    cells_by_id = {c.get('id'): c for c in all_cells}

    # ── Pass 1: collect all vertex positions to compute bounding box ──────────
    vertices = [c for c in all_cells
                if c.get('vertex') == '1' and c.get('id') not in ('0', '1')]
    edges    = [c for c in all_cells if c.get('edge') == '1']

    if not vertices:
        return

    xs, ys, x2s, y2s = [], [], [], []
    for v in vertices:
        ax, ay, aw, ah = resolve_geometry(v.get('id'), cells_by_id)
        if aw > 0 and ah > 0:
            xs.append(ax); ys.append(ay)
            x2s.append(ax + aw); y2s.append(ay + ah)

    if not xs:
        return

    # Content bounding box
    bb_x = min(xs);  bb_y = min(ys)
    bb_w = max(x2s) - bb_x
    bb_h = max(y2s) - bb_y

    # Scale to fit inside slide with margins
    avail_w = SLIDE_W_EMU - 2 * MARGIN
    avail_h = SLIDE_H_EMU - 2 * MARGIN
    scale_x = avail_w / bb_w if bb_w > 0 else 1
    scale_y = avail_h / bb_h if bb_h > 0 else 1
    sc = min(scale_x, scale_y)   # uniform scale

    def to_emu_x(v): return MARGIN + int((v - bb_x) * sc)
    def to_emu_y(v): return MARGIN + int((v - bb_y) * sc)
    def to_emu_w(v): return max(int(v * sc), 1)
    def to_emu_h(v): return max(int(v * sc), 1)

    # ── Pass 2: draw vertex shapes ────────────────────────────────────────────
    shape_map = {}   # cell_id → pptx shape (for connector endpoints)

    for v in vertices:
        vid = v.get('id')
        ax, ay, aw, ah = resolve_geometry(vid, cells_by_id)
        if aw <= 0 or ah <= 0:
            continue

        ex = to_emu_x(ax); ey = to_emu_y(ay)
        ew = to_emu_w(aw); eh = to_emu_h(ah)

        sd = parse_style(v.get('style', ''))
        prst = get_shape_type(sd)

        fill = hex_color(sd.get('fillColor'), RGBColor(0xFF,0xFF,0xFF))
        stroke = hex_color(sd.get('strokeColor'), RGBColor(0x66,0x66,0x66))
        font_color = hex_color(sd.get('fontColor'), RGBColor(0,0,0))
        dashed = sd.get('dashed') == '1'
        opacity = float(sd.get('opacity', 100))

        # Adjust fill for opacity
        if fill and opacity < 100:
            # blend with white
            r = int(fill[0] + (255 - fill[0]) * (1 - opacity/100))
            g = int(fill[1] + (255 - fill[1]) * (1 - opacity/100))
            b = int(fill[2] + (255 - fill[2]) * (1 - opacity/100))
            fill = RGBColor(r, g, b)

        # Font
        raw_font_size = float(sd.get('fontSize', 12))
        font_size = max(raw_font_size * sc / (914400 / 72 / 96), 6)
        # simpler: scale the font size relative to a reference 12pt at 1.0 scale
        # just use a fixed scaled font size
        fs = max(6.0, raw_font_size)   # fontSize in draw.io is already pt

        bold = sd.get('fontStyle', '0') in ('1', '3', '5', '7')

        val = v.get('value', '')
        text = text_from_value(val)

        # Swimlane: draw container rect + header text
        is_swimlane = 'swimlane' in sd
        start_size = float(sd.get('startSize', 20))

        if is_swimlane:
            # Header bar
            hdr_h = to_emu_h(start_size)
            hdr = make_shape(slide, 'rect', ex, ey, ew, hdr_h, fill, stroke, dashed=dashed)
            set_text(hdr, text, font_size=fs, bold=bold, font_color=font_color)
            # Body outline
            body = make_shape(slide, 'rect', ex, ey + hdr_h, ew, eh - hdr_h,
                              RGBColor(0xF8,0xF8,0xF8), stroke, dashed=dashed)
            shape_map[vid] = hdr
        else:
            sh = make_shape(slide, prst, ex, ey, ew, eh, fill, stroke,
                            opacity=100, dashed=dashed)
            align = PP_ALIGN.CENTER
            if 'align=left' in (v.get('style') or ''):
                align = PP_ALIGN.LEFT
            set_text(sh, text, font_size=fs, bold=bold, font_color=font_color, align=align)
            shape_map[vid] = sh

    # ── Pass 3: draw edges (orthogonal routing) ───────────────────────────────
    def geo_of(cid):
        return resolve_geometry(cid, cells_by_id)   # (ax, ay, aw, ah) in draw.io units

    def side_pt(cid, side):
        ax, ay, aw, ah = geo_of(cid)
        return _border_pt(ax, ay, aw, ah, side, to_emu_x, to_emu_y)

    def auto_sides(src, tgt):
        """Pick exit/entry side automatically from relative positions."""
        sx, sy, sw2, sh2 = geo_of(src)
        tx, ty, tw,  th  = geo_of(tgt)
        sc_cx, sc_cy = sx + sw2/2, sy + sh2/2
        tg_cx, tg_cy = tx + tw/2,  ty + th/2
        dy, dx = tg_cy - sc_cy, tg_cx - sc_cx
        if abs(dy) >= abs(dx):
            return ('bottom' if dy > 0 else 'top',
                    'top'    if dy > 0 else 'bottom')
        else:
            return ('right' if dx > 0 else 'left',
                    'left'  if dx > 0 else 'right')

    def abs_pt(geo_elem, attr, parent_id):
        """Absolute EMU for a named mxPoint, resolving parent offset."""
        if geo_elem is None:
            return None
        pt = geo_elem.find('mxPoint[@as="' + attr + '"]')
        if pt is None:
            return None
        px, py = float(pt.get('x', 0)), float(pt.get('y', 0))
        if parent_id not in ('0', '1'):
            ox, oy, _, _ = geo_of(parent_id)
            px += ox; py += oy
        return to_emu_x(px), to_emu_y(py)

    def parse_waypoints(geo_elem, parent_id):
        """Return list of explicit intermediate waypoints (EMU) from draw.io Array."""
        pts = []
        if geo_elem is None:
            return pts
        arr = geo_elem.find('Array[@as="points"]')
        if arr is None:
            return pts
        for mp in arr.findall('mxPoint'):
            px, py = float(mp.get('x', 0)), float(mp.get('y', 0))
            if parent_id not in ('0', '1'):
                ox, oy, _, _ = geo_of(parent_id)
                px += ox; py += oy
            pts.append((to_emu_x(px), to_emu_y(py)))
        return pts

    for e in edges:
        src_id = e.get('source')
        tgt_id = e.get('target')
        par_id = e.get('parent', '1')
        esd    = parse_style(e.get('style', ''))
        stroke = hex_color(esd.get('strokeColor'), RGBColor(0x33, 0x33, 0x33))
        dashed = esd.get('dashed') == '1'
        sw     = float(esd.get('strokeWidth', 1.5))
        geo    = e.find('mxGeometry')

        # ── Determine exit side & source point ───────────────────────────────
        if src_id and src_id in cells_by_id:
            raw_ex, raw_ey = esd.get('exitX'), esd.get('exitY')
            if raw_ex is not None and raw_ey is not None:
                exit_side = _side_from_xy(raw_ex, raw_ey)
            else:
                exit_side = auto_sides(src_id, tgt_id)[0] if (tgt_id and tgt_id in cells_by_id) else 'bottom'
            x1, y1 = side_pt(src_id, exit_side)
        else:
            pt = abs_pt(geo, 'sourcePoint', par_id)
            if pt is None:
                continue
            x1, y1 = pt
            exit_side = 'bottom'   # arbitrary; no source shape

        # ── Determine entry side & target point ───────────────────────────────
        if tgt_id and tgt_id in cells_by_id:
            raw_enx, raw_eny = esd.get('entryX'), esd.get('entryY')
            if raw_enx is not None and raw_eny is not None:
                entry_side = _side_from_xy(raw_enx, raw_eny)
            else:
                entry_side = auto_sides(src_id, tgt_id)[1] if (src_id and src_id in cells_by_id) else 'top'
            x2, y2 = side_pt(tgt_id, entry_side)
        else:
            pt = abs_pt(geo, 'targetPoint', par_id)
            if pt is None:
                continue
            x2, y2 = pt
            entry_side = 'top'

        # ── Build orthogonal path ─────────────────────────────────────────────
        explicit_wps = parse_waypoints(geo, par_id)
        if explicit_wps:
            # draw.io provided explicit waypoints → honour them, snap to orthogonal
            pts = [(x1, y1)] + explicit_wps + [(x2, y2)]
            # Snap each segment to H/V by extending the dominant axis
            snapped = [pts[0]]
            for i in range(1, len(pts)):
                px0, py0 = snapped[-1]
                px1, py1 = pts[i]
                if abs(px1 - px0) > abs(py1 - py0):
                    snapped.append((px1, py0))   # horizontal first
                else:
                    snapped.append((px0, py1))   # vertical first
                snapped.append((px1, py1))
            pts = snapped
        else:
            pts = _route(x1, y1, exit_side, x2, y2, entry_side)
            pts = _fix_exit(pts, x1, y1, exit_side)
            pts = _fix_approach(pts, x2, y2, entry_side)

        _draw_polyline(slide, pts, stroke_rgb=stroke, dashed=dashed,
                       width_pt=min(sw + 0.5, 2.5))

    # ── Slide title ───────────────────────────────────────────────────────────
    page_name = diagram_elem.get('name', '')
    if page_name:
        txb = slide.shapes.add_textbox(
            MARGIN, int(MARGIN * 0.3),
            SLIDE_W_EMU - 2 * MARGIN, int(MARGIN * 0.8)
        )
        tf = txb.text_frame
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = page_name
        run.font.bold = True
        run.font.size = Pt(13)
        run.font.color.rgb = RGBColor(0x1F,0x39,0x64)


# ── Entry point ───────────────────────────────────────────────────────────────
def convert(drawio_path, pptx_path):
    with open(drawio_path, 'r', encoding='utf-8') as f:
        content = f.read()

    root = ET.fromstring(content)
    diagrams = root.findall('diagram')

    prs = Presentation()
    prs.slide_width  = SLIDE_W_EMU
    prs.slide_height = SLIDE_H_EMU

    for i, diag in enumerate(diagrams):
        name = diag.get('name', f'Page {i+1}')
        print(f'  Converting: {name}')
        diagram_to_slide(diag, prs)

    prs.save(pptx_path)
    print(f'Saved: {pptx_path}')


if __name__ == '__main__':
    src = r'C:\Users\admin\Desktop\未命名绘图.drawio'
    dst = r'C:\Users\admin\Desktop\未命名绘图_WPS可编辑.pptx'
    convert(src, dst)
