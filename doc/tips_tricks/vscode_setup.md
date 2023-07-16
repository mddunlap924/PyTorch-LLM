# VSCode Setup
VSCode provides robust setup for projects. Listed below are some of links to relevant VSCode documentation.
- [User and Workspace Settings](https://code.visualstudio.com/docs/getstarted/settings)
- [Environment Variables](https://code.visualstudio.com/docs/python/environments#_environment-variables)
- [Python settings reference](https://code.visualstudio.com/docs/python/settings-reference)

# HuggingFace Cache Setup and Offline Mode
HuggingFace (HF) can be setup to download pretrained models to specified locations as well as setup to run in offline mode. Please refer to the HF website to read about the [Cache Setup](https://huggingface.co/docs/transformers/installation#cache-setup) and [Offline mode](https://huggingface.co/docs/transformers/installation#offline-mode).

To implement both the Cache setup and Offline mode the following was added in the project's folder:
### Step 1. 

Create a `prod.env` file at the same directory level as the `${workspaceFolder}`. Inside this file the following was written:

```env
# prod.env - production configuration

# HF cache setup
TRANSFORMERS_CACHE=/PATH/WHERE/HF/WILL/CACHE/OBJECTS

# HF offline mode
TRANSFORMERS_OFFLINE=1

# HF Parallelism 
TOKENIZERS_PARALLELISM=True
```

### Step 2. 

Inside the `.vscode` folder located at the same directory level as the file from Step 1 open the `settings.json` file.  Set the variable `python.envFile` as shown in the below snippet. Additional [Python settings reference](https://code.visualstudio.com/docs/python/settings-reference) can also be setup.

```json
"python.envFile": "${workspaceFolder}/prod.env",
```