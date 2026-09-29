$ErrorActionPreference = 'Stop'
foreach ($tool in @('ffmpeg', 'ffprobe')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "Required tool not found: $tool"
    }
}

$videoExtensions = @('.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v')
$imageExtensions = @('.jpg', '.jpeg', '.png', '.jfif', '.webp', '.bmp')
foreach ($candidate in Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'data') -Directory) {
    $files = @(Get-ChildItem -LiteralPath $candidate.FullName -File)
    $videos = @($files | Where-Object { $_.Extension.ToLowerInvariant() -in $videoExtensions })
    $images = @($files | Where-Object { $_.Extension.ToLowerInvariant() -in $imageExtensions })
    if ($videos.Count -eq 0 -or $images.Count -eq 0) { continue }
    if ($videos.Count -ne 1) { throw "Expected one video in $($candidate.Name)." }

    $durationText = & ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $videos[0].FullName
    if ($LASTEXITCODE -ne 0) { throw "Duration lookup failed for $($candidate.Name)." }
    $duration = [double]::Parse(($durationText -join '').Trim(), [Globalization.CultureInfo]::InvariantCulture)
    if ($duration -le 0) { throw "Invalid duration for $($candidate.Name)." }
    $frameDirectory = Join-Path $PSScriptRoot "outputs/$($candidate.Name)/frames"
    New-Item -ItemType Directory -Path $frameDirectory -Force | Out-Null
    foreach ($percent in @(20, 50, 80)) {
        $timestamp = ($duration * $percent / 100).ToString('0.######', [Globalization.CultureInfo]::InvariantCulture)
        $destination = Join-Path $frameDirectory "frame_$percent.jpg"
        & ffmpeg -hide_banner -loglevel error -y -ss $timestamp -i $videos[0].FullName -map 0:v:0 -frames:v 1 -update 1 -q:v 2 $destination
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $destination)) {
            throw "Frame extraction failed for $($candidate.Name) at $percent%."
        }
        Write-Output "$($candidate.Name): saved frame_$percent.jpg"
    }
}
