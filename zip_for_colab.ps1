# Run from: c:\Users\ACER\old_pc_data_dump2\llm-finetuning\
# Creates llm-finetuning.zip ready to upload to Google Drive

$source = "c:\Users\ACER\old_pc_data_dump2\llm-finetuning"
$dest   = "c:\Users\ACER\old_pc_data_dump2\llm-finetuning.zip"

# Remove old zip if exists
if (Test-Path $dest) { Remove-Item $dest }

# Files to include
$include = @(
    "data\processed\train.jsonl",
    "data\processed\val.jsonl",
    "data\processed\test.jsonl",
    "data\create_dataset.py",
    "training\finetune.py",
    "training\config.yaml",
    "training\colab_notebook.ipynb",
    "evaluation\metrics.py",
    "evaluation\evaluate.py",
    "evaluation\__init__.py",
    "inference\run_model.py",
    "inference\compare.py",
    "inference\__init__.py",
    "app.py",
    "requirements.txt"
)

$files = $include | ForEach-Object { Join-Path $source $_ } | Where-Object { Test-Path $_ }
Compress-Archive -Path $files -DestinationPath $dest

Write-Host "✅ Created: $dest"
Write-Host "📁 Upload llm-finetuning.zip to Google Drive (My Drive root)"
