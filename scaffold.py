import os

base_path = os.path.expanduser("~/Documents/Projects/kuro-bot-frontend/docs-processing")
project_structure = {
    "app": ["__init__.py", "main.py", "routes.py", "processor.py", "utils.py"],
    "": ["requirements.txt", "README.md", ".gitignore", "run.sh"]
}

# Create folders and files
for folder, files in project_structure.items():
    dir_path = os.path.join(base_path, folder)
    os.makedirs(dir_path, exist_ok=True)
    for file_name in files:
        file_path = os.path.join(dir_path, file_name)
        with open(file_path, "w") as f:
            f.write("")

print("✅ Scaffolded FastAPI project at", base_path)
