from __future__ import annotations

import sys

from run_vkrmd_vk_pipeline import main


if "--tasks" not in sys.argv:
    sys.argv += ["--tasks", "hpe"]

if __name__ == "__main__":
    main()
