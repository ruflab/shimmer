from migrate_ckpt import CkptType


def handle(ckpt: CkptType) -> CkptType:
    new_state_dict = {}
    for name, val in ckpt["state_dict"].items():
        if "loss_coefs.buffer" in name:
            continue
        if name[:12] == "domain_mods.":
            name = "gw_mod." + name
        if name[:18] == "gw_mod.domain_mods":
            new_state_dict["loss_mod." + name] = val
        new_state_dict[name] = val
    ckpt["state_dict"] = new_state_dict
    return ckpt
