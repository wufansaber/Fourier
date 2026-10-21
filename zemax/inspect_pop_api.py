"""Read-only introspection of the installed OpticStudio POP API."""

from __future__ import annotations

from pathlib import Path

from zosapi_utils import ZOSApplication


def public_names(value):
    return sorted(name for name in dir(value) if not name.startswith("_"))


def main() -> None:
    with ZOSApplication() as zos:
        api = zos.ZOSAPI
        system = zos.system
        sample = Path(zos.application.SamplesDir) / "Physical Optics" / "Basic Propagation.zos"
        system.LoadFile(str(sample), False)

        print(f"license={zos.application.LicenseStatus}")
        print(f"application_type={type(zos.application)}")
        print("application_members=" + ",".join(public_names(zos.application)))
        print(f"sample={sample}")
        print("analyses=" + ",".join(public_names(system.Analyses)))
        print("analysis_ids=" + ",".join(public_names(api.Analysis.AnalysisIDM)))

        pop = system.Analyses.New_Analysis(
            api.Analysis.AnalysisIDM.PhysicalOpticsPropagation
        )
        settings = pop.GetSettings()
        print(f"pop_type={type(pop)}")
        print(f"settings_type={type(settings)}")
        print("pop_members=" + ",".join(public_names(pop)))
        print("settings_members=" + ",".join(public_names(settings)))

        for name in (
            "StartSurface",
            "EndSurface",
            "Wavelength",
            "Field",
            "BeamType",
            "XSampling",
            "YSampling",
            "XWidth",
            "YWidth",
            "TotalPower",
            "NumberOfParameters",
        ):
            try:
                value = getattr(settings, name)
                print(f"setting.{name}={value!r};type={type(value)}")
            except Exception as exc:
                print(f"setting.{name}=ERROR:{exc}")
        for index in range(settings.NumberOfParameters):
            print(
                f"parameter[{index}]={settings.GetParameterName(index)!r}:"
                f"{settings.GetParameterValue(index)!r}"
            )
        print("start_surface_members=" + ",".join(public_names(settings.StartSurface)))
        print("end_surface_members=" + ",".join(public_names(settings.EndSurface)))
        original_beam_type = settings.BeamType
        for beam_type in range(0, 8):
            try:
                settings.BeamType = beam_type
                settings.Update()
                parameters = [
                    (settings.GetParameterName(i), settings.GetParameterValue(i))
                    for i in range(settings.NumberOfParameters)
                ]
                print(f"beam_type[{beam_type}]={parameters!r}")
            except Exception as exc:
                print(f"beam_type[{beam_type}]=ERROR:{exc}")
        settings.BeamType = original_beam_type
        settings.Update()
        print("surface_apertures=" + ",".join(public_names(api.Editors.LDE.SurfaceApertureTypes)))
        print("surface_types=" + ",".join(public_names(api.Editors.LDE.SurfaceType)))
        aperture_data = system.LDE.GetSurfaceAt(1).ApertureData
        print("aperture_data_members=" + ",".join(public_names(aperture_data)))
        user_aperture = aperture_data.CreateApertureTypeSettings(
            api.Editors.LDE.SurfaceApertureTypes.UserAperture
        )
        print("user_aperture_members=" + ",".join(public_names(user_aperture)))
        print(
            "user_aperture_specific_members="
            + ",".join(public_names(user_aperture._S_UserAperture))
        )
        pop.Close()


if __name__ == "__main__":
    main()
