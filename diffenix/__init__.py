from __future__ import annotations
import numpy as np
import sys
import os
from diffenix.Constantes import *

if __name__ == '__main__':

    from diffenix.time_integration import time_integration, restart
    if len(sys.argv) > 1:
        import os
        if sys.argv[1] == "configs":
            l = os.listdir("./configs")
            l.sort()
            if int(sys.argv[2]) < len(l):
                params = dict(np.load("./configs/" + l[int(sys.argv[2])]))
                print(int(sys.argv[2]), l[int(sys.argv[2])], params["Simulation"])
                if not os.path.exists(results + "/" + params["Simulation"]):
                    time_integration(consts_params=params)
                # else:
                #     restart(params["Simulation"], rtol=np.double(1e-4))
        else:
            print("argv :", sys.argv[1])
            restart(int(sys.argv[1]), -1)
    else:
        time_integration()