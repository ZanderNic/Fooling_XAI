# Fooling_XAI

# installation
Install uv according to [this doc](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer):
> macOS and Linux
> `curl -LsSf https://astral.sh/uv/install.sh | sh`

> Windows 
> `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

Then create the venv and downlaod packages using `uv sync`. Then just activate teh enviroment or use it in vscode etc..

To add new packages use `uv add <name>`, just like pip but using "add" instead of install. Same for uninstall/remove.