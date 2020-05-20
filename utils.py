from decimal import Decimal
import math


class status_bar:
    """
    A simple status bar for loops.
    """
    def __init__(self, estimated_reps, bar_size=40):
        """
        Parameters
        ----------
        estimated_reps: int
            The number of expected times this status bar will be updated (usually the number of loops).
            
        bar_size: int
            The character size of the status bar (make this smaller for small screens). Default 40.
            
        
        Usage
        -----
        Initiate this as an object and call its .start method before running the loop. Then once every loop iteration call its .iterate method. Additional text can be added to the printout by inputing it to the .iterate method every call.
        
        Example usage:
        > import time
        > n_loops = 10
        > sb = status_bar(n_loops)
        > sb.start()
        > for i in range(n_loops):
            > time.sleep(.2)
            > sb.iterate()
        """
        self.reps = estimated_reps
        self.bar_size = bar_size
        self.finished = 1
        self.ratio = bar_size/estimated_reps
        
    def start(self):
        print('-'*(self.bar_size-self.finished), end="\r")
        
    def iterate(self, text=None):
        finished = int(self.ratio*self.finished)
        if(finished>self.bar_size):
            print('='*self.bar_size+' {cr}/{er} (estimated reps exceeded!)  {txt}'.format(
                er=self.reps, cr=self.finished, txt=text if text!=None else ''
            ), end="\r")
        else:
            print('{eqs}{mins} {fin}/{reps}'.format(
                eqs = '='*finished,
                mins = '-'*(self.bar_size-finished),
                fin = str(self.finished),
                reps = str(self.reps)
            ), end="\r")
#             print('='*finished+'-'*(self.bar_size-finished)+' '+str(self.finished)+'/'+str(self.reps), end="\r")
        self.finished += 1





def sig_fig(num, sd):
#     num = float(num) if type(num)==str else num
    num = round(num, -len(str(num))+sd)
    val = '%.'+str(sd-1)+'E'
    return(val % Decimal(str(num)))
