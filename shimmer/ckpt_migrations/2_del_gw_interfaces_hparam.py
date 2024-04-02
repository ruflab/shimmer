from migrate_ckpt import CkptType


def handle(ckpt: CkptType) -> CkptType:
    if "hyper_parameters" in ckpt and "gw_interfaces" in ckpt["hyper_parameters"]:
        del ckpt["hyper_parameters"]["gw_interfaces"]
    return ckpt
