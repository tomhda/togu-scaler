; TOGU SCALER のセットアップ。scripts\build_installer.ps1 から作る。
; 利用者ごとのインストール（管理者権限なし）で、既定の場所は
; %LOCALAPPDATA%\Programs\togu-scaler。

#define AppVersion GetEnv("TOGU_APP_VERSION")
#define PublishDir GetEnv("TOGU_PUBLISH_DIR")
#define RepoDir GetEnv("TOGU_REPO_DIR")

#if AppVersion == ""
  #error "TOGU_APP_VERSION must be set by scripts/build_installer.ps1"
#endif
#if PublishDir == ""
  #error "TOGU_PUBLISH_DIR must be set by scripts/build_installer.ps1"
#endif
#if RepoDir == ""
  #error "TOGU_REPO_DIR must be set by scripts/build_installer.ps1"
#endif

[Setup]
AppId={{2D4F5A69-8E9A-43A4-B7A4-ED88126CBDA5}
AppName=TOGU SCALER
AppVersion={#AppVersion}
AppVerName=TOGU SCALER {#AppVersion}
AppPublisher=TOGU SCALER
AppPublisherURL=https://github.com/tomhda/togu-scaler
AppSupportURL=https://github.com/tomhda/togu-scaler/issues
AppUpdatesURL=https://github.com/tomhda/togu-scaler/releases
VersionInfoVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\togu-scaler
DisableProgramGroupPage=yes
UninstallDisplayName=TOGU SCALER
UninstallDisplayIcon={app}\togu-scaler.exe
SetupIconFile={#RepoDir}\app\assets\app.ico
PrivilegesRequired=lowest
ChangesEnvironment=yes
SetupArchitecture=x64
ArchitecturesAllowed=x64compatible
MinVersion=10.0.19041
WizardStyle=modern
Compression=lzma2/normal
SolidCompression=yes
CloseApplications=yes
OutputBaseFilename=togu-scaler-setup-win64

[Languages]
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
japanese.AdditionalTasks=追加タスク:
japanese.DesktopIcon=デスクトップにショートカットを作る
japanese.AddToPath=togu-scaler-cli を名前だけで実行できるよう、フォルダーを PATH に追加する（スクリプトや AI エージェント向け）
japanese.LaunchApp=TOGU SCALER を起動する
english.AdditionalTasks=Additional tasks:
english.DesktopIcon=Create a desktop shortcut
english.AddToPath=Add the folder to PATH, to start togu-scaler-cli by name (for scripts and AI agents)
english.LaunchApp=Launch TOGU SCALER

[Tasks]
Name: "desktopicon"; Description: "{cm:DesktopIcon}"; GroupDescription: "{cm:AdditionalTasks}"; Flags: unchecked
Name: "addtopath"; Description: "{cm:AddToPath}"; GroupDescription: "{cm:AdditionalTasks}"; Flags: unchecked

[InstallDelete]
; 前の版のライブラリが残らないよう、本体の _internal は入れ直す。
; 追加キット（vendor・models・tools の中）と NPU の変換結果はここでは消さない。
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#PublishDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\TOGU SCALER"; Filename: "{app}\togu-scaler.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\TOGU SCALER"; Filename: "{app}\togu-scaler.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\togu-scaler.exe"; Description: "{cm:LaunchApp}"; Flags: postinstall nowait skipifsilent

[UninstallDelete]
; 後から展開した追加キットと、アプリが作る NPU の変換結果も一緒に外す。
; models\custom（利用者が置いたモデル）は残す。
Type: filesandordirs; Name: "{app}\vendor\amd-npu-1.8"
Type: filesandordirs; Name: "{app}\vendor\winml-film"
Type: filesandordirs; Name: "{app}\models\ai"
Type: filesandordirs; Name: "{app}\models\film"
Type: filesandordirs; Name: "{app}\tools"
Type: files; Name: "{app}\*キットの使い方.txt"
Type: files; Name: "{app}\How to use the * kit.txt"
Type: dirifempty; Name: "{app}\vendor"
Type: dirifempty; Name: "{app}\models"
Type: dirifempty; Name: "{app}"

[Code]
// 「addtopath」を選んだときだけ、インストール先を利用者の PATH に足す。
// アンインストールで同じものを外す。PATH のほかの項目には触れない。

const
  EnvironmentKey = 'Environment';

function SameFolder(const A, B: string): Boolean;
begin
  Result := CompareText(RemoveBackslashUnlessRoot(Trim(A)), RemoveBackslashUnlessRoot(Trim(B))) = 0;
end;

// Paths から Folder と同じ項目を除いたものを返す。見つかったかを Found に入れる。
function WithoutFolder(const Paths, Folder: string; var Found: Boolean): string;
var
  Rest, Entry: string;
  Separator: Integer;
begin
  Result := '';
  Found := False;
  Rest := Paths;
  while Rest <> '' do
  begin
    Separator := Pos(';', Rest);
    if Separator = 0 then
    begin
      Entry := Rest;
      Rest := '';
    end
    else
    begin
      Entry := Copy(Rest, 1, Separator - 1);
      Rest := Copy(Rest, Separator + 1, Length(Rest));
    end;

    if Trim(Entry) = '' then
      Continue;

    if SameFolder(Entry, Folder) then
      Found := True
    else
    begin
      if Result <> '' then
        Result := Result + ';';
      Result := Result + Entry;
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Paths, Remaining: string;
  Found: Boolean;
begin
  if (CurStep <> ssPostInstall) or not WizardIsTaskSelected('addtopath') then
    Exit;

  if not RegQueryStringValue(HKCU, EnvironmentKey, 'Path', Paths) then
    Paths := '';
  Remaining := WithoutFolder(Paths, ExpandConstant('{app}'), Found);
  if Found then
    Exit;

  if Remaining <> '' then
    Remaining := Remaining + ';';
  RegWriteExpandStringValue(HKCU, EnvironmentKey, 'Path', Remaining + ExpandConstant('{app}'));
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Paths, Remaining: string;
  Found: Boolean;
begin
  if CurUninstallStep <> usUninstall then
    Exit;

  if not RegQueryStringValue(HKCU, EnvironmentKey, 'Path', Paths) then
    Exit;
  Remaining := WithoutFolder(Paths, ExpandConstant('{app}'), Found);
  if Found then
    RegWriteExpandStringValue(HKCU, EnvironmentKey, 'Path', Remaining);
end;
