import sys
import os
import argparse

if sys.version_info.major < 3:
    from Tkinter import Tk
    from Tkinter import IntVar
    from Tkinter import Checkbutton
    from Tkinter import Button
    from Tkinter import Label
    from Tkinter import mainloop
    from Tkinter import W
    from tkFileDialog import askopenfilename
else:
    from tkinter import Tk
    from tkinter import IntVar
    from tkinter import Checkbutton
    from tkinter import Button
    from tkinter import Label
    from tkinter import mainloop
    from tkinter import W
    from tkinter.filedialog import askopenfilename

try:
    from . import haussio
except (ValueError, SystemError):
    import haussio


THOR_RAW = "Image_0001_0001.raw"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "rawfile", type=argparse.FileType('r'), nargs='?',
        help="Raw file or directory name")
    parser.add_argument(
        "--mp", help="Convert to multipage tiff", action="store_true",
        default=True)
    return parser.parse_args()


def raw2tiff(rawfile, mp):
    if not os.path.exists(rawfile):
        rawfile = os.path.join(rawfile, THOR_RAW)
        if not os.path.exists(rawfile):
            sys.stderr.write("Could not find {0}\n".format(
                os.path.abspath(args.rawfile)))
            quit(1)

    xml_file = os.path.join(
        os.path.abspath(os.path.dirname(rawfile)), "Experiment.xml")
    if not os.path.exists(xml_file):
        sys.stderr.write("Could not find {0}\n".format(
            xml_file))
        quit(1)

    sys.stdout.write("Converting...\n")

    data_haussio = haussio.ThorHaussIO(
        os.path.abspath(os.path.dirname(rawfile)), width_idx=5)
    data_haussio.raw2tiff(mp=mp)
    sys.stdout.write("done\n")


def gui():
    master = Tk()

    rawfile = askopenfilename(
        title="Select raw file", filetypes=[("Raw files", "*.raw")])

    Label(master, text="Multipage TIFF").grid(row=0, sticky=W)
    var = IntVar()
    Checkbutton(master, text="Generate multipage TIFF", variable=var).grid(
        row=1, sticky=W)
    Button(master, text='Cancel', command=quit).grid(
        row=2, sticky=W, pady=4)
    Button(master, text='Convert', command=master.quit).grid(
        row=3, sticky=W, pady=4)
    mainloop()
    mp = var.get() != 0

    return rawfile, mp


if __name__ == "__main__":
    args = parse_arguments()
    if args.rawfile is None:
        rawfile, mp = gui()
    else:
        rawfile = os.path.abspath(args.rawfile)
        mp = args.mp

    raw2tiff(rawfile, mp)
