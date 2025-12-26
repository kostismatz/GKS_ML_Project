if (-not (Test-Path -Path ".\fixed")) {
    Write-Host "Folder structure not found! creating..."
    1..10 | ForEach-Object 
    {
        New-Item -ItemType Directory -Path ".\fixed\fold$_" -Force | Out-Null
    }
    Write-Host "done"
}


$files = Get-ChildItem -Path "fold*" -Filter "*.wav" -Recurse

foreach ($file in $files) 
{
    $subFolder = $file.Directory.Name
    $fileName = $file.Name
    $outputPath = Join-Path -Path ".\fixed" -ChildPath (Join-Path -Path $subFolder -ChildPath $fileName)
    
    ffmpeg -i "$($file.FullName)" -ar 16000 "$outputPath"
}