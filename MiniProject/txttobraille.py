def toBraille(c):
unic=2800
mapping = " A1B'K2L@CIF/MSP\"E3H9O6R^DJG>NTQ,*5<-U8V.%[$+X!&;:4\\0Z7(_?W]#Y)="
i = mapping.index(c.upper())
if (i>0):
    unic+=i 
    unichex = hex(unic)
    return unichr(unichex))
if (i==0):
    return '_'
if (i<O):
    return '?'

def converter(txt):
tmp=""
for x in txt:
    tmp+=str(toBraille(x))
return tmp

txt = raw_input("Please insert text: \n")
print(converter(txt))
