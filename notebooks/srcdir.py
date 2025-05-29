import pathlib
import sys


srcdir = (pathlib.Path.cwd() / '../src').resolve()
if str(srcdir) not in sys.path:
    sys.path.append(str(srcdir))
    print(f'{srcdir} is added into sys.path')
