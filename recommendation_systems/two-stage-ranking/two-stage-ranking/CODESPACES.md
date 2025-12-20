# GitHub Codespaces Quick Start

This project is configured to run in **GitHub Codespaces** - a cloud-based development environment.

## 🚀 Launch in Codespaces

1. Go to: https://github.com/ferrosas2/helloworld
2. Click the **Code** button (green)
3. Select the **Codespaces** tab
4. Click **"Create codespace on master"**

GitHub will automatically:
- ✅ Set up Python 3.11 environment
- ✅ Install all dependencies from `requirements.txt`
- ✅ Install AWS CLI and Docker
- ✅ Configure VS Code with Python extensions

## ⚙️ Configure AWS Credentials

Once your Codespace is running, set up AWS access:

```bash
# Set environment variables
export AWS_ACCESS_KEY_ID=your_access_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_access_key
export AWS_DEFAULT_REGION=us-east-1

# Or use AWS configure
aws configure
```

## 🏃 Run the Project

```bash
# Navigate to project directory
cd recommendation_systems/two-stage-ranking

# Train the model (replace with your S3 bucket)
python src/train.py \
  --bucket ltr-models-frp \
  --key data/ltr_training_data.csv \
  --output-dir ./models

# Test inference
python src/inference.py --model-path ./models/model.json

# Run Jupyter notebook
jupyter notebook notebooks/ltr.ipynb
```

## 🐳 Build Docker Container

```bash
# Build
docker build -t two-stage-ranking:latest .

# Run
docker run --rm \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  -v $(pwd)/models:/opt/ml/model \
  two-stage-ranking:latest \
  --bucket your-bucket \
  --key data/file.csv
```

## 💡 Tips

- **Free Tier**: GitHub provides 60 hours/month of Codespaces for free
- **Port Forwarding**: Jupyter (8888) and API (8080) are auto-forwarded
- **Persistence**: Your workspace is saved between sessions
- **Secrets**: Add AWS credentials as [Codespaces secrets](https://github.com/settings/codespaces) for security

## 🔗 Resources

- [GitHub Codespaces Docs](https://docs.github.com/en/codespaces)
- [Managing Secrets](https://docs.github.com/en/codespaces/managing-your-codespaces/managing-secrets-for-your-codespaces)
