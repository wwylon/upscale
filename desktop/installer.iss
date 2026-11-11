; Inno Setup — produces a single Upscale-Setup-cpu.exe users can download and run.
; Prerequisites:
;   1. py -3 -m PyInstaller desktop/upscale-cpu.spec --noconfirm
;   2. Install Inno Setup 6, then compile this script
;
; Output: desktop\Output\Upscale-Setup-cpu.exe

#define MyAppName "Upscale"
; AppVersion is hardcoded — keep it equal to [project].version in pyproject.toml.
#define MyAppVersion "0.5.0"
#define MyAppPublisher "Upscale"
#define MyAppExeName "Upscale.exe"

[Setup]
AppId={{A7C0E5D2-9F41-4B8E-9C2A-UPSCALE2026}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\Upscale
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=Output
OutputBaseFilename=Upscale-Setup-cpu
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "polish"; MessagesFile: "compiler:Languages\Polish.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Additional icons:"

[Files]
Source: "..\dist\Upscale\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
; "Open with Upscale" for common image types (per-user)
Root: HKCU; Subkey: "Software\Classes\Applications\{#MyAppExeName}\shell\open\command"; ValueType: string; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Flags: uninsdeletekey

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch Upscale"; Flags: nowait postinstall skipifsilent
