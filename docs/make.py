from pathlib import Path

import pdoc

modules = [
    "shimmer.types",
    "shimmer.modules.global_workspace",
    "shimmer.modules.domain",
    "shimmer.modules.gw_module",
    "shimmer.modules.losses",
    "shimmer.modules.contrastive_loss",
    "shimmer.dataset",
    "shimmer.modules.vae",
    "shimmer.modules.utils",
    "shimmer.utils",
]

here = Path(__file__).parent

if __name__ == "__main__":
    pdoc.render.configure(docformat="google", math=True)
    pdoc.pdoc(*modules, output_directory=here / "api")
