# Script to push two-stage-ranking folder to public GitHub repo
# This preserves the existing git configuration

# Configuration
$PUBLIC_REPO = "https://github.com/ferrosas2/helloworld.git"
$REMOTE_NAME = "public-repo"
$SUBTREE_PATH = "AWS/recommendation_systems/two-stage-ranking"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Pushing two-stage-ranking to public repo" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Navigate to repository root
Set-Location "c:\Users\DELL\OneDrive\public-repos\RAG\RAG-Ingestion"

# Check if remote already exists
$remoteExists = git remote | Select-String -Pattern $REMOTE_NAME -Quiet

if (-not $remoteExists) {
    Write-Host "`n[1/4] Adding remote '$REMOTE_NAME'..." -ForegroundColor Yellow
    git remote add $REMOTE_NAME $PUBLIC_REPO
    Write-Host "Remote added successfully" -ForegroundColor Green
} else {
    Write-Host "`n[1/4] Remote '$REMOTE_NAME' already exists" -ForegroundColor Green
}

# Fetch the public repo to see its current state
Write-Host "`n[2/4] Fetching public repo..." -ForegroundColor Yellow
git fetch $REMOTE_NAME

# Push the subtree to the public repo
Write-Host "`n[3/4] Pushing subtree to public repo..." -ForegroundColor Yellow
Write-Host "Source: $SUBTREE_PATH" -ForegroundColor Gray

# Create a temporary branch with just the subtree content
Write-Host "Creating temporary branch from subtree..." -ForegroundColor Gray
git subtree split --prefix=$SUBTREE_PATH -b temp-two-stage-ranking

if ($LASTEXITCODE -eq 0) {
    Write-Host "Subtree split successful" -ForegroundColor Green
    
    # Push the temporary branch to the public repo master branch
    Write-Host "Pushing to public repo master branch..." -ForegroundColor Gray
    git push $REMOTE_NAME temp-two-stage-ranking:master --force
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully pushed to public repo!" -ForegroundColor Green
    } else {
        Write-Host "Push failed. Check your permissions and repo URL." -ForegroundColor Red
    }
    
    # Delete the temporary branch
    Write-Host "Cleaning up temporary branch..." -ForegroundColor Gray
    git branch -D temp-two-stage-ranking
    Write-Host "Cleanup complete!" -ForegroundColor Green
} else {
    Write-Host "Subtree split failed." -ForegroundColor Red
}

# Verify remotes
Write-Host "`n[4/4] Current remotes:" -ForegroundColor Yellow
git remote -v

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Done! Your local repo sync is intact." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
