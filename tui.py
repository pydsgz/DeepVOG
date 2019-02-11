import deepvog as dv
import sys

if __name__ == "__main__":
    if len(sys.argv)>1:

        tui = dv.tui(str(sys.argv[1]))
        tui.run_tui()
