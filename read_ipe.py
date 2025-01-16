"""
Takes all vertices of path objects, and scales them
into the square [0,1]x[0,1].

Matrix attributes are ignored!
If you have translated or scaled certain parts,
cut and paste all objects once before saving the file.
(No, that does not help.)
"""

def rescale(p):
    'into square of side length 0.9, centered in unit square'
    x0 = min(x for x,y in p)
    x1 = max(x for x,y in p)
    y0 = min(y for x,y in p)
    y1 = max(y for x,y in p)
    dx = x1-x0
    dy = y1-y0
    d = max(dx,dy)
    scale = 0.9/d
    ax = 0.5 - scale*(x0+x1)/2 # (x0+x1)/2 is mapped to 0.5
    ay = 0.5 - scale*(y0+y1)/2 # (x0+x1)/2 is mapped to 0.5
    return [x*scale+ax for x,y in p], [y*scale+ay for x,y in p]

def read_ipe(fname):
    p = set()
    with open(fname) as ipefile:
        started = False
        for l in ipefile:
            if started:
                s = l.split()
                if len(s)==3:
                    x,y,m = s
                    if m in ("m","l"):
                        p.add((float(x),float(y)))
            else:
                started = l=="<page>\n"
    p = list(p)
    print(len(p),"points read from file",fname)
    x,y = rescale(p)
    return x,y
            
#read_ipe("bad-sizes.ipe")
