# -*- mode: python ; coding: utf-8 -*-
from os.path import join as pjoin
import os
import sys
import periodictable
import refnx
import pip

block_cipher = None


refnx_version = refnx.version.version

periodictable_loc = os.path.dirname(periodictable.__file__)
pip_loc = os.path.dirname(pip.__file__)

pathex = pjoin(".", "")
uiloc = (
    pjoin("..", "..", "refnx", "reflect", "_app", "ui", "*.ui"),
    pjoin("refnx", "reflect", "_app", "ui"),
)
icons = (
    pjoin("..", "..", "refnx", "reflect", "_app", "icons", "*.png"),
    pjoin("refnx", "reflect", "_app", "icons"),
)
licences = (
    pjoin("..", "..", "refnx", "reflect", "_app", "ui", "licences", "*"),
    pjoin("refnx", "reflect", "_app", "ui", "licences"),
)
lipid_data = (
    pjoin("..", "..", "refnx", "reflect", "_app", "lipids.json"),
    pjoin("refnx", "reflect", "_app"),
)
periodic_table = (
    pjoin(periodictable_loc, "xsf", "*.*"),
    pjoin("periodictable", "xsf"),
)

a = Analysis(
    ['motofit.py'],
    pathex=[],
    binaries=[],
    datas=[uiloc, icons, licences, lipid_data, periodic_table],
    hiddenimports=[
        "pkg_resources",
        "periodictable",
        "black",
        "refnx",
        "refnx.analysis",
        "refnx.dataset",
        "refnx.reflect",
        "refnx.reflect._app",
        "attrs",
        "corner",
        "pint",
        "pkg_resources",
        "scipy.special.cython_special",
        "scipy.spatial.transform._rotation_groups",
        "refnx.reflect._app",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

# OSX
if sys.platform == "darwin":
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='refnx',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon="../../refnx/reflect/_app/icons/Motofit.icns",

    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='refnx',
    )
    app = BUNDLE(
        coll,
        name='refnx.app',
        icon="../../refnx/reflect/_app/icons/Motofit.icns",
                info_plist={
                "CFBundleName": "refnx",
                "CFBundleIdentifier": "com.refnx.refnx",
                "NSPrincipalClass": "NSApplication",
                "NSHighResolutionCapable": "True",
                "CFBundleShortVersionString": refnx_version,
                "LSBackgroundOnly": "False",
                "CFBundleDocumentTypes": [
                    {
                        "CFBundleTypeName": "refnx experiment file",
                        "CFBundleTypeExtensions": ("mtft",),
                        "CFBundleTypeIconFile": "Motofit.icns",
                    }
                ],
            },
    )

# windows

elif sys.platform in ["win32", "cygwin"]:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        name="motofit",
        bootloader_ignore_signals=False,
        debug=False,
        strip=False,
        upx=False,
        console=False,
        icon="..\\..\\refnx\\reflect\\_app\\icons\\scattering.ico",
    )
