from pytorch_pipeline.utils.system import resolve_hardware_profile


def test_resolve_hardware_profile():
    profile = resolve_hardware_profile()
    print(profile)
