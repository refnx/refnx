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
    ["motofit.py"],
    pathex=[os.getcwd()],
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
        "pkg_resources",
        "scipy.special.cython_special",
        "scipy.spatial.transform._rotation_groups",
        "refnx.reflect._app",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["pyqt6"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)


# OSX
if sys.platform == "darwin":
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        name="motofit",
        debug=False,
        strip=False,
        upx=True,
        console=False,
        icon="../../refnx/reflect/_app/icons/Motofit.icns",
    )

    app = BUNDLE(
        exe,
        name="motofit.app",
        icon="../../refnx/reflect/_app/icons/Motofit.icns",
        bundle_identifier=None,
        info_plist={
            "CFBundleName": "motofit",
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

"""
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='motofit',
          debug=False,
          bootloader_ignore_signals=False,
          strip=True,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=True,
               upx=True,
               name='motofit')
"""
