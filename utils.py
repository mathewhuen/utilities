from decimal import Decimal
import math

def sigfig(num, sd):
#     num = float(num) if type(num)==str else num
    num = round(num, -len(str(num))+sd)
    val = '%.'+str(sd-1)+'E'
    return(val % Decimal(str(num)))
