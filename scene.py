"""Draws the live picture as an SVG: sky, sun, clouds, hills and the solar tracker.

Side view of the east-west plane. Angles run from the EAST horizon (0 deg, left) over the zenith
(90 deg) to the WEST horizon (180 deg, right). The sun is at `sun_deg`; the panel's normal points at
`panel_deg`; pointing error = sun - panel.
"""
import math

from simulation import cloud_factor

W, H = 960, 600                 # picture size
PX, PY = 480, 382               # pivot of the panel
R_SUN = 292                     # distance of the sun from the pivot
GROUND_Y = 405

# (dx, dy, size, side) of each cloud relative to the sun; side -1 = comes from the left, +1 = from the right
CLOUDS = [(-10, 5, 1.5, -1), (75, -15, 1.3, 1), (-85, -25, 1.2, -1), (40, 38, 1.1, 1),
          (-40, -55, 1.0, -1), (115, 12, 1.0, 1), (-125, 18, 0.9, -1), (10, -75, 0.9, 1)]
MAX_CLOUDS = len(CLOUDS)


def n_clouds(severity):
    return 0 if severity <= 0 else max(1, round(severity * MAX_CLOUDS))


def sun_position(angle_deg):
    a = math.radians(angle_deg)
    return PX - R_SUN * math.cos(a), PY - R_SUN * math.sin(a)


def mix(c1, c2, a):
    """Blend two '#rrggbb' colours: a=0 gives c1, a=1 gives c2."""
    a = min(max(a, 0.0), 1.0)
    rgb = lambda c: [int(c[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * a) for x, y in zip(rgb(c1), rgb(c2)))


def _cloud(cx, cy, size, fill, opacity):
    bumps = [(0, 0, 28), (-30, 8, 22), (30, 10, 24), (-55, 16, 16), (55, 18, 16), (10, -14, 22)]
    parts = [f'<g opacity="{opacity:.2f}">',
             f'<ellipse cx="{cx:.1f}" cy="{cy + 22 * size:.1f}" rx="{72 * size:.1f}" ry="{14 * size:.1f}" fill="{fill}"/>']
    parts += [f'<circle cx="{cx + dx * size:.1f}" cy="{cy + dy * size:.1f}" r="{r * size:.1f}" fill="{fill}"/>'
              for dx, dy, r in bumps]
    return parts + ["</g>"]


def _defs(sky_top, sky_bottom, cover):
    return [
        '<defs>',
        f'<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{sky_top}"/>'
        f'<stop offset="1" stop-color="{sky_bottom}"/></linearGradient>',
        f'<radialGradient id="glow"><stop offset="0" stop-color="#fff2a8" stop-opacity="{0.9 * (1 - 0.75 * cover):.2f}"/>'
        '<stop offset="1" stop-color="#ffd166" stop-opacity="0"/></radialGradient>',
        '<linearGradient id="grass" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#3f8f4a"/>'
        '<stop offset="1" stop-color="#276331"/></linearGradient>',
        '<marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">'
        '<path d="M0,0 L8,4 L0,8 z" fill="#4cc9f0"/></marker>',
        '</defs>',
        f'<rect width="{W}" height="{H}" fill="url(#sky)"/>']


def _sun(sx, sy, cover):
    path = " ".join("%.0f,%.0f" % sun_position(d) for d in range(0, 181, 3))
    return [f'<polyline points="{path}" fill="none" stroke="#ffffff" stroke-opacity="0.35" stroke-dasharray="6 8"/>',
            f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="95" fill="url(#glow)"/>',
            f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="34" fill="#ffd23f" opacity="{1 - 0.45 * cover:.2f}"/>']


def _clouds(t, sx, sy, severity, drift):
    """Clouds enter from the sides and gather over the sun; `drift` is 0 (off-screen) to 1 (over the sun)."""
    fill = mix("#ffffff", "#8e98a3", severity * 0.9)
    parts = []
    for i in range(n_clouds(severity)):
        dx, dy, size, side = CLOUDS[i]
        start = -220 - 70 * i if side < 0 else W + 220 + 70 * i
        x = start + (sx + dx - start) * drift + 10 * math.sin(0.35 * t + i)
        y = sy + dy + 4 * math.sin(0.5 * t + i)
        if -160 < x < W + 160:
            parts += _cloud(x, y, size, fill, 0.55 + 0.4 * severity)
    return parts


def _ground():
    ridge = (f'M0,{GROUND_Y} L0,372 C100,360 200,384 300,378 C380,372 420,394 480,394 C540,394 580,372 660,378 '
             f'C760,384 860,360 {W},372 L{W},{GROUND_Y} Z')
    return [f'<path d="{ridge}" fill="#2f6f4b"/>',
            f'<rect x="0" y="{GROUND_Y}" width="{W}" height="{H - GROUND_Y}" fill="url(#grass)"/>']


def _rays(sx, sy, ux, uy, cover):
    """Dotted sunlight lines from the sun to the panel; (ux, uy) is the unit vector from sun to panel."""
    parts = []
    for off in (-40, -20, 0, 20, 40):
        x1, y1 = sx + ux * 55 - uy * off, sy + uy * 55 + ux * off
        x2, y2 = PX - uy * off * 0.5, PY + ux * off * 0.5
        parts.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="#ffd166" '
                     f'stroke-width="2" stroke-dasharray="3 9" opacity="{0.55 * (1 - 0.85 * cover):.2f}"/>')
    return parts


def _tracker(panel_deg, cover, ux, uy):
    """Post, motor, rotating panel, and the two arrows (panel normal, direction to the sun)."""
    parts = [f'<line x1="{PX}" y1="{PY}" x2="{PX}" y2="485" stroke="#5b6168" stroke-width="9"/>',
             f'<ellipse cx="{PX}" cy="487" rx="46" ry="9" fill="#3b4045"/>',
             f'<rect x="{PX - 16}" y="{PY - 16}" width="32" height="32" rx="6" fill="#40464d" stroke="#8a929b"/>',
             f'<g transform="rotate({panel_deg - 90.0:.2f} {PX} {PY})">'
             f'<rect x="{PX - 90}" y="{PY - 7}" width="180" height="14" rx="2" fill="#1d3a6b" stroke="#a9c4f5" stroke-width="1.5"/>']
    parts += [f'<line x1="{PX - 90 + k * 18}" y1="{PY - 7}" x2="{PX - 90 + k * 18}" y2="{PY + 7}" stroke="#4f78b8" stroke-width="1"/>'
              for k in range(1, 10)]
    parts += [f'<rect x="{PX - 90}" y="{PY - 7}" width="180" height="3.5" fill="#79b0ff"/>'
              f'<circle cx="{PX + 84}" cy="{PY - 12}" r="4.5" fill="{mix("#ffe08a", "#7f8790", cover)}" stroke="#222"/></g>',
              f'<circle cx="{PX}" cy="{PY}" r="7" fill="#d8dde3"/>']
    a = math.radians(panel_deg)
    nx, ny = -math.cos(a), -math.sin(a)                                   # panel normal on the screen
    parts += [f'<line x1="{PX}" y1="{PY}" x2="{PX + 125 * nx:.1f}" y2="{PY + 125 * ny:.1f}" '
              'stroke="#4cc9f0" stroke-width="3" marker-end="url(#arr)"/>',
              f'<line x1="{PX}" y1="{PY}" x2="{PX - ux * 170:.1f}" y2="{PY - uy * 170:.1f}" stroke="#ffb703" '
              'stroke-width="2" stroke-dasharray="7 6"/>']
    return parts


def _hud(t, label, sun_deg, panel_deg, severity, cover, sensor):
    """Information strip at the bottom."""
    columns = [(40, [f"t = {t:5.1f} s", label, f"clouds: {n_clouds(severity)} | cover {cover * 100:.0f} %"]),
               (330, [f"Sun angle:      {sun_deg:7.2f} deg", f"Panel angle:    {panel_deg:7.2f} deg",
                      f"Pointing error: {sun_deg - panel_deg:7.2f} deg"]),
               (650, [f"Light-sensor signal: {sensor * 100:3.0f} %"])]
    parts = [f'<rect x="20" y="500" width="{W - 40}" height="88" rx="12" fill="#000" opacity="0.55"/>']
    for x0, lines in columns:
        for k, text in enumerate(lines):
            if text:
                parts.append(f'<text x="{x0}" y="{528 + 24 * k}" font-size="15" fill="#fff" font-family="monospace">{text}</text>')
    parts += ['<rect x="650" y="540" width="240" height="9" rx="4" fill="#fff" opacity="0.25"/>',
              f'<rect x="650" y="540" width="{240 * sensor:.0f}" height="9" rx="4" fill="{mix("#ff595e", "#7bd88f", sensor)}"/>',
              '<text x="650" y="566" font-size="13" fill="#4cc9f0" font-family="monospace">blue arrow = panel normal</text>',
              '<text x="650" y="583" font-size="13" fill="#ffd166" font-family="monospace">orange dashed = direction to sun</text>']
    return parts


def render_scene(t, sun_deg, panel_deg, severity, cloud_start, cloud_end, label=""):
    drift = cloud_factor(t, cloud_start, cloud_end) if severity > 0 else 0.0     # how far the clouds have arrived
    cover = severity * drift                      # 0..1: share of the sunlight that is lost
    sensor = 1.0 - 0.9 * cover                    # light-sensor signal strength
    sx, sy = sun_position(sun_deg)
    d = math.hypot(PX - sx, PY - sy)
    ux, uy = (PX - sx) / d, (PY - sy) / d         # unit vector from the sun to the panel

    # Sky colour follows the sun's height above the horizon and turns grey under clouds.
    height = min(max(min(sun_deg, 180.0 - sun_deg), 0.0) / 35.0, 1.0)
    sky_top = mix(mix("#2d3f6e", "#3b82d6", height), "#6f7a86", 0.6 * cover)
    sky_bottom = mix(mix("#f4a261", "#bfe3ff", height), "#aab2bb", 0.6 * cover)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Arial, sans-serif">']
    parts += _defs(sky_top, sky_bottom, cover)
    parts += _sun(sx, sy, cover)
    parts += _clouds(t, sx, sy, severity, drift)
    parts += _ground()                            # hills hide the lower half of the rising / setting sun
    parts += _rays(sx, sy, ux, uy, cover)
    parts += _tracker(panel_deg, cover, ux, uy)
    parts += _hud(t, label, sun_deg, panel_deg, severity, cover, sensor)
    return "".join(parts + ["</svg>"])