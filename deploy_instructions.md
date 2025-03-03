# Deploying Qwen2.5-VL API to Render

This document provides step-by-step instructions for deploying the Qwen2.5-VL API to Render.

## Prerequisites

1. A [Render](https://render.com/) account
2. A GitHub account
3. A credit card for Render billing (Pro plan is required)

## Step 1: Prepare Repository

1. Create a new GitHub repository for hosting the API code
2. Copy all the files from the `qwen-render-api` folder to this repository:
   - `server.py` - The FastAPI server implementation
   - `requirements.txt` - Dependencies
   - `render.yaml` - Render configuration
   - `README.md` - Documentation

## Step 2: Deploy to Render

1. Log in to your [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" and select "Web Service"
3. Select "Deploy from GitHub repository"
4. Connect your GitHub account if you haven't already
5. Select the repository you created with the API code
6. Configure the following settings:
   - **Name**: `qwen-vl-api` (or your preferred name)
   - **Region**: Choose a region close to your users
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: Leave empty
   - **Runtime**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python server.py`
   - **Plan**: Pro (at least 8 GB RAM needed)

7. Scroll down to "Advanced" and add the following environment variables:
   - `MODEL_ID`: `Qwen/Qwen2.5-VL-3B-Instruct`
   - `USE_INT8`: `true`
   - `DEVICE`: `cpu`
   - `MAX_NEW_TOKENS`: `1024`
   - `PORT`: `8080`
   - `PYTHONUNBUFFERED`: `true`

8. Click "Create Web Service"

## Step 3: Add Persistent Disk

1. After the initial deploy, go to your web service settings
2. Navigate to "Disks" in the left sidebar
3. Click "Add Disk"
4. Configure the disk:
   - **Name**: `qwen-model-cache`
   - **Mount Path**: `/root/.cache`
   - **Size**: 50 GB (minimum)
5. Click "Save"

## Step 4: Set up CORS (if needed)

If you need to access the API from different domains:

1. Go to your web service settings
2. Navigate to "Environment" in the left sidebar
3. Add a variable `ALLOWED_ORIGINS` with a comma-separated list of domains, or `*` for all domains
4. Click "Save Changes"

## Step 5: Test Your Deployment

1. After deployment completes (which can take 15-30 minutes due to model downloading), your API will be available at `https://your-service-name.onrender.com`
2. Test the health endpoint: `https://your-service-name.onrender.com/health`
3. You should see a response like:
   ```json
   {"status": "healthy", "model_id": "Qwen/Qwen2.5-VL-3B-Instruct"}
   ```

## Step 6: Update VerifiedX to Use the API

1. Update your `config.py` file with your API URL:
   ```python
   QWEN_API_URL = "https://your-service-name.onrender.com"
   QWEN_USE_API = True
   ```

2. Run the system with the API client:
   ```bash
   python main.py --use-qwen-api
   ```

## Troubleshooting

- **Memory Issues**: If you see out-of-memory errors, make sure you've selected the Pro plan and added the persistent disk.
- **Slow Model Loading**: The first request will take time as the model is loaded. Subsequent requests will be faster.
- **Network Errors**: Check that the API URL is correct and the service is running.
- **API Authentication**: If you need to add authentication, modify the `server.py` file to include an API key check.