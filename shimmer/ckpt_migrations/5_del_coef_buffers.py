from migrate_ckpt import CkptType


def handle(ckpt: CkptType) -> CkptType:
    new_state_dict = {}
    for name, val in ckpt["state_dict"].items():
        if "coef_buffers." in name:
            continue
        new_state_dict[name] = val
    ckpt["state_dict"] = new_state_dict
    return ckpt
