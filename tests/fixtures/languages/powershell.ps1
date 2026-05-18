param(
    [Parameter(Mandatory = $false)]
    [string]$Root = "."
)

$items = Get-ChildItem -Path $Root -Filter *.rs -Recurse |
    Where-Object { -not $_.PSIsContainer } |
    Sort-Object FullName

foreach ($item in $items) {
    $hash = Get-FileHash -Algorithm SHA256 -Path $item.FullName
    Write-Host ("{0} {1}" -f $hash.Hash, $item.FullName)
}

if ($items.Count -eq 0) {
    Write-Warning "no rust files found"
}
