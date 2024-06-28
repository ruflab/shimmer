import sys
from pathlib import Path

import pdoc
from jinja2 import Environment, FileSystemLoader, select_autoescape

modules = [
    "shimmer.types",
    "shimmer.modules.global_workspace",
    "shimmer.modules.domain",
    "shimmer.modules.gw_module",
    "shimmer.modules.selection",
    "shimmer.modules.losses",
    "shimmer.modules.contrastive_loss",
    "shimmer.dataset",
    "shimmer.modules.vae",
    "shimmer.utils",
    "shimmer.cli.ckpt_migration",
]

here = Path(__file__).parent

if __name__ == "__main__":
    args = sys.argv[1:]
    version = "latest"
    if len(args):
        version = args[0]

    pdoc.render.configure(docformat="google", math=True)
    pdoc.pdoc(*modules, output_directory=here / "api" / version)

    env = Environment(
        loader=FileSystemLoader(here),
        autoescape=select_autoescape(),
        comment_start_string="{=",
        comment_end_string="=}",
    )
    template = env.get_template("index.html.jinja2")

    latest_version: None | str = None
    doc_versions: list[str] = []
    for folder in (here / "api").iterdir():
        if folder.name == "index.html" or folder.name == ".gitignore":
            continue
        if folder.name != "latest":
            doc_versions.append(folder.name)
        else:
            latest_version = folder.name
    doc_versions = list(sorted(doc_versions, reverse=True))
    if latest_version is not None:
        doc_versions = [latest_version] + doc_versions

    with open(here / "api" / "index.html", "w") as f:
        f.write(template.render(versions=doc_versions))
