# Setting API Keys

This guide explains how to set the `GEMINI_API_KEY` environment variable for the Streamlit app.

## Option 1: Set in Terminal (Temporary - Current Session Only)

```bash
# Activate your virtual environment first
source venv_sam/bin/activate

# Set the API key
export GEMINI_API_KEY="your-api-key-here"

# Then run Streamlit
streamlit run sam_refine_ui.py
```

**Note:** This only works for the current terminal session. If you close the terminal, you'll need to set it again.

## Option 2: Set in Shell Profile (Permanent)

Add the export command to your shell profile so it's set automatically:

### For Bash (default on macOS/Linux):
```bash
# Edit your ~/.bashrc or ~/.bash_profile
nano ~/.bashrc

# Add this line (replace with your actual API key):
export GEMINI_API_KEY="your-api-key-here"

# Save and reload
source ~/.bashrc
```

### For Zsh (default on newer macOS):
```bash
# Edit your ~/.zshrc
nano ~/.zshrc

# Add this line (replace with your actual API key):
export GEMINI_API_KEY="your-api-key-here"

# Save and reload
source ~/.zshrc
```

## Option 3: Create a `.env` File (Recommended for Development)

1. Create a `.env` file in the repo root:
```bash
cd /path/to/imbot
nano .env
```

2. Add your API key:
```
GEMINI_API_KEY=your-api-key-here
```

3. The `.env` file is already in `.gitignore`, so it won't be committed to git.

**Note:** Streamlit doesn't automatically load `.env` files. You'll need to either:
- Use `python-dotenv` package and load it in the script, OR
- Source it manually: `export $(cat .env | xargs)` before running Streamlit

## Option 4: Set in Streamlit Config (Not Recommended)

You can also set it in Streamlit's config, but this is less secure and not recommended.

## Getting Your Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key (it starts with something like `AIza...`)

## Verifying It's Set

After setting the key, verify it's available:

```bash
# Check if it's set
echo $GEMINI_API_KEY

# Should output your API key (if set)
```

## In the Streamlit App

Once the API key is set, you'll see:
- ✅ "Gemini QC available" in the sidebar
- 🔍 "Run Gemini QC" button becomes active

If not set, you'll see:
- 💡 "Set GEMINI_API_KEY to enable QC"

## Security Note

**Never commit your API key to git!** The `.env` file and any files containing API keys are already in `.gitignore`.

