from migrate_ckpt import CkptType


def handle(ckpt: CkptType) -> CkptType:
    new_state_dict = {}
    for name, val in ckpt["state_dict"].items():
        if "gw_mod.gw_interfaces" in name and "domain_module" in name:
            continue
        elif "gw_mod.gw_interfaces" in name and "encoder" in name:
            new_name = name.replace(".gw_interfaces", ".gw_encoders")
            new_name = new_name.replace(".encoder", "")
            new_state_dict[new_name] = val
        elif "gw_mod.gw_interfaces" in name and "decoder" in name:
            new_name = name.replace(".gw_interfaces", ".gw_decoders")
            new_name = new_name.replace(".decoder", "")
            new_state_dict[new_name] = val
        elif "gw_interfaces" in name:
            print(name)
        else:
            new_state_dict[name] = val
    ckpt["state_dict"] = new_state_dict
    return ckpt
