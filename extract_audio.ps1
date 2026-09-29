param(
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$Only
)

$ErrorActionPreference = 'Stop'
$ffmpeg = (Get-Command ffmpeg -ErrorAction Stop).Source
$ffprobe = (Get-Command ffprobe -ErrorAction Stop).Source

$folders = if ($Only) {
    Get-Item -LiteralPath (Join-Path (Join-Path $PSScriptRoot 'data') $Only)
} else {
    Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'data') -Directory
}
$folders | ForEach-Object {
    $folder = $_
    try {
        $video = Join-Path $folder.FullName 'video.mp4'
        if (-not (Test-Path -LiteralPath $video -PathType Leaf)) {
            $videos = @(Get-ChildItem -LiteralPath $folder.FullName -Filter '*.mp4' -File)
            if ($videos.Count -ne 1) {
                throw 'Expected video.mp4 or exactly one MP4 file.'
            }
            $video = $videos[0].FullName
        }
        $destination = Join-Path (Join-Path $PSScriptRoot 'outputs') $folder.Name
        New-Item -ItemType Directory -Path $destination -Force | Out-Null
        $audio = Join-Path $destination 'audio.wav'
        & $ffmpeg -hide_banner -loglevel error -nostdin -y -i $video -vn -ac 1 -ar 16000 -c:a pcm_s16le $audio
        if ($LASTEXITCODE -ne 0) { throw "ffmpeg failed (exit code $LASTEXITCODE)." }
        $duration = & $ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $audio
        if ($LASTEXITCODE -ne 0) { throw 'Could not read extracted audio duration.' }
        Write-Host "$($folder.Name): $duration seconds -> $audio"
    }
    catch {
        Write-Host "FAILED $($folder.Name): $($_.Exception.Message)"
    }
}
