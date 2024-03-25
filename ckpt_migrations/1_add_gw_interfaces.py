from migrate_ckpt import CkptType


def handle(ckpt: CkptType) -> CkptType:
    new_state_dict = {}
    for name, val in ckpt["state_dict"].items():
        new_name = name.replace(
            "gw_mod.encoders.resnet", "gw_mod.gw_interfaces.resnet.encoder"
        )
        new_name = new_name.replace(
            "gw_mod.encoders.bge", "gw_mod.gw_interfaces.bge.encoder"
        )
        new_name = new_name.replace(
            "gw_mod.decoders.resnet", "gw_mod.gw_interfaces.resnet.decoder"
        )
        new_name = new_name.replace(
            "gw_mod.decoders.bge", "gw_mod.gw_interfaces.bge.decoder"
        )
        new_state_dict[new_name] = val
    ckpt["state_dict"] = new_state_dict
    return ckpt
