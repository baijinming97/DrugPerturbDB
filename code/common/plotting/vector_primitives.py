class Drawing:
    """Emit identical vector primitives into an SVG group and a PDF."""

    def __init__(self, height):
        self.height = height
        self.group = ET.Element(tag('g'), {'id': 'refined_schematic'})
        self.buffer = io.BytesIO()
        self.pdf = canvas.Canvas(self.buffer, pagesize=(W * SCALE, height * SCALE), pageCompression=1)
        self.pdf.scale(SCALE, SCALE)

    def color(self, color, stroke=False):
        rgb = [int(color[i:i + 2], 16) / 255 for i in [1, 3, 5]]
        (self.pdf.setStrokeColorRGB if stroke else self.pdf.setFillColorRGB)(*rgb)

    def rect(self, x, y, w, h, fill=FILL, stroke=BORDER, r=0, lw=0.6):
        a = {'x': str(x), 'y': str(y), 'width': str(w), 'height': str(h), 'fill': fill, 'stroke': stroke, 'stroke-width': str(lw)}
        if r:
            a.update(rx=str(r), ry=str(r))
        ET.SubElement(self.group, tag('rect'), a)
        self.color(fill)
        self.color(stroke, True)
        self.pdf.setLineWidth(lw)
        if r:
            self.pdf.roundRect(x, self.height - y - h, w, h, r, fill=1, stroke=1)
        else:
            self.pdf.rect(x, self.height - y - h, w, h, fill=1, stroke=1)

    def text(self, x, y, s, size=8, bold=False, color=INK, anchor='middle'):
        st = f'font-size: {size}px; font-family: DejaVu Sans; text-anchor: {anchor}; fill: {color}'
        if bold:
            st += '; font-weight: 700'
        ET.SubElement(self.group, tag('text'), {'x': str(x), 'y': str(y), 'style': st}).text = s
        self.color(color)
        self.pdf.setFont('DejaVu Sans Bold' if bold else 'DejaVu Sans', size)
        fn = self.pdf.drawCentredString if anchor == 'middle' else self.pdf.drawRightString if anchor == 'end' else self.pdf.drawString
        fn(x, self.height - y, s)

    def arrow(self, pts, color=LINE, lw=0.7, head=3):
        p = self.pdf.beginPath()
        p.moveTo(pts[0][0], self.height - pts[0][1])
        for x, y in pts[1:]:
            p.lineTo(x, self.height - y)
        self.color(color, True)
        self.pdf.setLineWidth(lw)
        self.pdf.drawPath(p)
        ET.SubElement(self.group, tag('polyline'), {'points': ' '.join((f'{x},{y}' for x, y in pts)), 'fill': 'none', 'stroke': color, 'stroke-width': str(lw), 'stroke-linejoin': 'round'})
        x, y = pts[-1]
        px, py = pts[-2]
        dx = x - px
        dy = y - py
        ln = (dx * dx + dy * dy) ** 0.5
        dx /= ln
        dy /= ln
        tri = [(x, y), (x - head * dx + head * 0.5 * dy, y - head * dy - head * 0.5 * dx), (x - head * dx - head * 0.5 * dy, y - head * dy + head * 0.5 * dx)]
        ET.SubElement(self.group, tag('polygon'), {'points': ' '.join((f'{xx},{yy}' for xx, yy in tri)), 'fill': color})
        p = self.pdf.beginPath()
        p.moveTo(tri[0][0], self.height - tri[0][1])
        for xx, yy in tri[1:]:
            p.lineTo(xx, self.height - yy)
        p.close()
        self.color(color)
        self.pdf.drawPath(p, fill=1, stroke=0)

    def finish(self):
        self.pdf.showPage()
        self.pdf.save()
        return fitz.open(stream=self.buffer.getvalue(), filetype='pdf')
