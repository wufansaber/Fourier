"""Small ZOS-API bootstrap helpers for OpticStudio 2022 R2.02."""

from __future__ import annotations

import os
import winreg

import clr


OPTICSTUDIO_PATH = r"C:\Program Files\Ansys Zemax OpticStudio 2022 R2.02"


class ZOSApplication:
    """Connect to OpticStudio and close only owned standalone instances."""

    def __init__(self, install_path: str = OPTICSTUDIO_PATH) -> None:
        try:
            with winreg.OpenKey(
                winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER),
                r"Software\Zemax",
                0,
                winreg.KEY_READ,
            ) as key:
                zemax_root = winreg.QueryValueEx(key, "ZemaxRoot")[0]
            net_helper = os.path.join(
                zemax_root, "ZOS-API", "Libraries", "ZOSAPI_NetHelper.dll"
            )
        except FileNotFoundError:
            # A repaired/reinstalled OpticStudio can leave an obsolete
            # per-user NetHelper behind while removing HKCU\Software\Zemax.
            # Prefer the helper shipped with the explicitly selected install;
            # mixing (for example) a 2017 helper with a 2022 installation can
            # stall CreateNewApplication without raising a managed exception.
            candidates = (
                os.path.join(install_path, "ZOSAPI_NetHelper.dll"),
                os.path.join(
                    os.path.expanduser("~"), "Documents", "Zemax",
                    "ZOS-API", "Libraries", "ZOSAPI_NetHelper.dll"
                ),
            )
            net_helper = next(
                (candidate for candidate in candidates
                 if os.path.isfile(candidate)), None
            )
            if net_helper is None:
                raise FileNotFoundError(
                    "No ZOSAPI_NetHelper.dll was found for OpticStudio"
                )
        clr.AddReference(net_helper)
        import ZOSAPI_NetHelper

        if not ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize(install_path):
            raise RuntimeError(f"Cannot initialize OpticStudio at {install_path}")
        zos_dir = ZOSAPI_NetHelper.ZOSAPI_Initializer.GetZemaxDirectory()

        clr.AddReference(os.path.join(zos_dir, "ZOSAPI.dll"))
        clr.AddReference(os.path.join(zos_dir, "ZOSAPI_Interfaces.dll"))
        import ZOSAPI

        self.ZOSAPI = ZOSAPI
        self.connection = ZOSAPI.ZOSAPI_Connection()
        self._owns_application = (
            os.environ.get("ZOSAPI_CONNECT_AS_EXTENSION", "0") != "1"
        )
        if self._owns_application:
            self.application = self.connection.CreateNewApplication()
        else:
            self.application = self.connection.ConnectAsExtension(0)
        if self.application is None:
            raise RuntimeError("ZOS-API failed to create an OpticStudio application")
        if not self.application.IsValidLicenseForAPI:
            # Do not leave a headless OpticStudio process behind when a login
            # or license refresh is required.
            if self._owns_application:
                self.application.CloseApplication()
            self.application = None
            raise RuntimeError("The installed OpticStudio license is not valid for ZOS-API")
        self.system = self.application.PrimarySystem
        if self.system is None:
            raise RuntimeError("ZOS-API did not return a primary optical system")

    def close(self) -> None:
        if (getattr(self, "application", None) is not None
                and self._owns_application):
            self.application.CloseApplication()
            self.application = None

    def __enter__(self) -> "ZOSApplication":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
