"""python -m crawlers 入口"""

import sys
from pathlib import Path

# 确保项目根目录在path中
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from crawlers.cli import cli

if __name__ == "__main__":
    cli()
