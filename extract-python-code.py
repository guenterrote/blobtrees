import re

startpattern = re.compile(r'\\startlisting\{(.*)\}')

inListing = False
numListing = 0
with (open("theprogram.tex", 'r') as infile,
      open("blobtree-extracted.py", 'w') as outfile):
    for line in infile:
        m = startpattern.match(line)
        if m:
            numListing += 1
            title = m.group(1)
            titletext = f"***** Listing {numListing}: {title} *****"
            if numListing>1:
                print(file=outfile) # empty line for separation
            print("#","*"*len(titletext), file=outfile)
            print("#",titletext, file=outfile)
            print("#","*"*len(titletext)+"\n", file=outfile)
            inListing = True
        elif line.strip().startswith(r'\end{lstlisting}'):
            inListing = False
        elif inListing:
            outfile.write((line))
print("File blobtree-extracted.py written.")
