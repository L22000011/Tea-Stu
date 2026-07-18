$ErrorActionPreference = "Continue"

$Root = "E:\Deskbook\Tea\Raw-RGB"
$Output = Join-Path $Root "MMFi_RGB_SUBSET"
$Downloader = Join-Path $Root "download_mmfi_rgb_subset.py"
$LogDir = Join-Path $Root "download_logs"
$TargetCount = 64152
$ChunkSize = 1000

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Get-PngCount {
    if (-not (Test-Path $Output)) {
        return 0
    }
    return (Get-ChildItem -Path $Output -Recurse -Filter *.png | Measure-Object).Count
}

while ($true) {
    $count = Get-PngCount
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $log = Join-Path $LogDir "continue_20pct_$stamp.log"

    "[$(Get-Date)] Current PNG count: $count / $TargetCount" | Tee-Object -FilePath $log -Append
    if ($count -ge $TargetCount) {
        "[$(Get-Date)] Target reached. Stop." | Tee-Object -FilePath $log -Append
        break
    }

    $remaining = $TargetCount - $count
    $thisChunk = [Math]::Min($ChunkSize, $remaining)
    "[$(Get-Date)] Downloading next chunk: $thisChunk new files" | Tee-Object -FilePath $log -Append

    & python $Downloader `
        --preset global-percent `
        --percent 20 `
        --max-new-files $thisChunk `
        --progress-every 250 `
        --show-first 3 `
        --retries 5 `
        --retry-sleep 5 2>&1 | Tee-Object -FilePath $log -Append

    Start-Sleep -Seconds 10
}
