# Script to properly add two-stage-ranking to public repo
# This adds it to recommendation_systems/ folder WITHOUT overwriting existing files

$PUBLIC_REPO = "https://github.com/ferrosas2/helloworld.git"
$REMOTE_NAME = "public-repo"
$TEMP_DIR = "$env:TEMP\helloworld-temp"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Adding two-stage-ranking to /recommendation_systems/" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Clean up any previous temp directory
if (Test-Path $TEMP_DIR) {
    Remove-Item -Path $TEMP_DIR -Recurse -Force
}

# Clone the public repo
Write-Host "`n[1/5] Cloning public repo..." -ForegroundColor Yellow
git clone $PUBLIC_REPO $TEMP_DIR
Set-Location $TEMP_DIR

# Create recommendation_systems folder if it doesn't exist
Write-Host "`n[2/5] Ensuring recommendation_systems/ folder exists..." -ForegroundColor Yellow
if (-not (Test-Path "recommendation_systems")) {
    New-Item -ItemType Directory -Path "recommendation_systems" | Out-Null
    Write-Host "Created recommendation_systems/ folder" -ForegroundColor Green
} else {
    Write-Host "recommendation_systems/ folder already exists" -ForegroundColor Green
}

# Copy the two-stage-ranking folder
Write-Host "`n[3/5] Copying two-stage-ranking folder..." -ForegroundColor Yellow
$SOURCE = "c:\Users\DELL\OneDrive\public-repos\RAG\RAG-Ingestion\AWS\recommendation_systems\two-stage-ranking"
Copy-Item -Path $SOURCE -Destination "recommendation_systems\two-stage-ranking" -Recurse -Force
Write-Host "Files copied successfully" -ForegroundColor Green

# Add and commit
Write-Host "`n[4/5] Committing changes..." -ForegroundColor Yellow
git add recommendation_systems/two-stage-ranking/
git commit -m "Add two-stage-ranking LTR system to recommendation_systems/"

# Push to master
Write-Host "`n[5/5] Pushing to public repo..." -ForegroundColor Yellow
git push origin master

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nSuccessfully added to recommendation_systems/!" -ForegroundColor Green
    Write-Host "View at: https://github.com/ferrosas2/helloworld/tree/master/recommendation_systems/two-stage-ranking" -ForegroundColor Cyan
} else {
    Write-Host "`nPush failed!" -ForegroundColor Red
}

# Clean up
Write-Host "`nCleaning up temp directory..." -ForegroundColor Gray
Set-Location c:\Users\DELL\OneDrive\public-repos\RAG\RAG-Ingestion
Remove-Item -Path $TEMP_DIR -Recurse -Force

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Done! Your existing files are preserved." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
