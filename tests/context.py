import sys, os
from pathlib import Path

current_folder = Path(os.path.dirname(os.path.abspath(__file__)))
source = current_folder.parent
package_folder = os.path.join(source, "pythhon3d")
sys.path.insert(0, package_folder)
