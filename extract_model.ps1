# Run after Drive zip finishes downloading
# Extracts model adapters to outputs/model/

$project = "c:\Users\ACER\old_pc_data_dump2\llm-finetuning"
$downloads = "C:\Users\ACER\Downloads"

# Find the downloaded zip (Drive downloads as llm-finetuning-outputs.zip or similar)
$zip = Get-ChildItem $downloads -Filter "*.zip" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if (-not $zip) {
    Write-Host "No zip found in Downloads. Wait for Chrome to finish downloading."
    exit 1
}

Write-Host "Found: $($zip.FullName)"

$dest = "$project\outputs"
New-Item -ItemType Directory -Force -Path "$dest\model" | Out-Null
New-Item -ItemType Directory -Force -Path "$dest\results" | Out-Null

Expand-Archive -Path $zip.FullName -DestinationPath "$dest\extracted_temp" -Force

# Move adapter files to outputs/model/
$modelSrc = Get-ChildItem "$dest\extracted_temp" -Recurse -Filter "adapter_config.json" |
            Select-Object -First 1

if ($modelSrc) {
    $modelDir = $modelSrc.DirectoryName
    Copy-Item "$modelDir\*" "$dest\model\" -Recurse -Force
    Write-Host "✅ Model adapters → outputs/model/"
}

# Move evaluation.csv if present
$csvSrc = Get-ChildItem "$dest\extracted_temp" -Recurse -Filter "evaluation.csv" |
          Select-Object -First 1
if ($csvSrc) {
    Copy-Item $csvSrc.FullName "$dest\results\" -Force
    Write-Host "✅ evaluation.csv → outputs/results/"
}

Remove-Item "$dest\extracted_temp" -Recurse -Force
Write-Host ""
Write-Host "Contents of outputs/model/:"
Get-ChildItem "$dest\model" | ForEach-Object { Write-Host "  $($_.Name)" }
