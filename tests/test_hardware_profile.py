from plant_pheno.utils.system import resolve_hardware_profile


def test_resolve_hardware_profile():
    profile = resolve_hardware_profile()
    print(profile)
