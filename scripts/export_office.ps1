# Open the report (Word) and the deck (PowerPoint) in real Microsoft Office, export PDFs, record pages/slides.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\export_office.ps1 [-SlidePngDir <dir>]
param([string]$SlidePngDir = "")
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$results = @()
$word = New-Object -ComObject Word.Application
$word.Visible = $false; $word.DisplayAlerts = 0
try {
    foreach ($f in Get-ChildItem (Join-Path $root 'reports') -Filter *.docx) {
        $doc = $null
        try {
            $doc = $word.Documents.Open($f.FullName, $false, $true)
            $pages = $doc.ComputeStatistics(2)
            $doc.SaveAs2([IO.Path]::ChangeExtension($f.FullName, '.pdf'), 17)
            $results += [pscustomobject]@{ file = $f.Name; opened = $true; details = "pages=$pages" }
        } catch { $results += [pscustomobject]@{ file = $f.Name; opened = $false; details = $_.Exception.Message } }
        finally { if ($doc) { $doc.Close(0) } }
    }
} finally { $word.Quit(); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word) }
$pp = New-Object -ComObject PowerPoint.Application
try {
    foreach ($f in Get-ChildItem (Join-Path $root 'presentation') -Filter *.pptx -ErrorAction SilentlyContinue) {
        $pres = $null
        try {
            $pres = $pp.Presentations.Open($f.FullName, -1, 0, 0)
            $n = $pres.Slides.Count
            $pres.SaveAs([IO.Path]::ChangeExtension($f.FullName, '.pdf'), 32)
            if ($SlidePngDir) {
                New-Item -ItemType Directory -Force $SlidePngDir | Out-Null
                for ($i = 1; $i -le $n; $i++) { $pres.Slides.Item($i).Export((Join-Path $SlidePngDir ("slide{0:D2}.png" -f $i)), 'PNG', 1600, 900) }
            }
            $results += [pscustomobject]@{ file = $f.Name; opened = $true; details = "slides=$n" }
        } catch { $results += [pscustomobject]@{ file = $f.Name; opened = $false; details = $_.Exception.Message } }
        finally { if ($pres) { $pres.Close() } }
    }
} finally { $pp.Quit(); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($pp) }
$results | ConvertTo-Json | Out-File -Encoding utf8 (Join-Path $root 'reports\office_open_check.json')
$results | Format-Table -AutoSize
if ($results | Where-Object { -not $_.opened }) { exit 1 }
