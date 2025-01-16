

from ipestart import ipestart

def open_ipe(fname="blob-tree.ipe"):
    global ipefile
    ipefile = open(fname,'w')
    ipefile.write(ipestart)
    return ipefile

def start_page():
    ipefile.write("<page>\n")
def end_page():
    ipefile.write("</page>\n")

def close_ipe():
    ipefile.write("</ipe>\n")
    ipefile.close()
    print("File",ipefile.name,"written.",
          'Run "ipetoipe -pdf '+ipefile.name+'" to generate a pdf file.')

def start_frame(unit=500,x=30,y=70):
    ipefile.write(
    """<group layer="alpha" matrix="%f 0 0 %f %f %f">
""" % (unit,unit,x,y))
    ipefile.write("""<path stroke="black">
0 1 m
0 0 l
1 0 l
1 1 l
h
</path>
""")

def end_frame():
    ipefile.write("</group>\n")

def put_text(text, x=30, y=30):
    r"text should use &quot; &lt; &gt; etc. \ is fine."
    ipefile.write('<text transformations="translations" pos="%f %f" stroke="black" '
                  'type="label" valign="baseline">%s</text>\n' % (x,y,text))

def edge(x1,y1,x2,y2,color=None,extras = ""):
    if not color:
        color = 'black'
    ipefile.write("""<path stroke="%s" %s>
%f %f m
%f %f l
</path>
""" % (color, extras, x1,y1,x2,y2))


arrow = ' arrow="normal/normal"'

dotform = '<use name="mark/disk(sx)" pos="%f %f" size="normal" stroke="black"/>\n'

    
############################
    
def draw_tree(x,y,succ, point_labels = True):
    for i,j in enumerate(succ):
        ipefile.write(dotform % (x[i],y[i]))
        if point_labels: put_text(str(i),x[i],y[i])
        if j is not None:
            edge(x[i],y[i],x[j],y[j],extras=arrow)    

    
def draw_edge(x,y,i,j,color=None, extras=""):
    edge(x[i],y[i],x[j],y[j],color, extras)    


def draw_blob(x,y,i,j,color=None): #....
    edge(x[i],y[i],x[j],y[j],color)    

    
    
