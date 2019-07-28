import re
import numpy as np


def reformat(xpath, ypath, outpath):
    xdata = []
    ydata = []
    with open(xpath, "r") as x:
        with open(ypath, "r") as y:
            for n, yline in enumerate(y):
                ymatch = re.search(r"-*\d*.?\d+", yline)
                if ymatch:
                    ydata.append(ymatch.group())
                    row = []
                    while True:
                        xline = x.readline()
                        if '----' in xline:
                            while True:
                                xline = x.readline()
                                xmatch = re.findall(r'-*\d.*?\d+', xline)
                                if len(xmatch) > 2:
                                    row.extend((xmatch[0], xmatch[1], xmatch[2]))
                                else:
                                    xdata.append(row)
                                    break
                            break

    np.savez(outpath, np.array(xdata, float), np.array(ydata, float))
    print("Success")


reformat("AllPositions.txt", "AllEnergies.txt", "formatted.npz")
