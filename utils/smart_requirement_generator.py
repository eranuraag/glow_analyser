# generate_smart_requirements.py
import pkg_resources
import ast
import os

def get_imports_from_code(directory):
    imports = set()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.add(alias.name.split('.')[0])
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imports.add(node.module.split('.')[0])
                except:
                    continue
    return imports

def generate_requirements():
    # Get imports from code
    code_imports = get_imports_from_code('.')
    
    # Get installed packages
    installed_packages = {pkg.project_name.lower(): pkg.version 
                         for pkg in pkg_resources.working_set}
    
    # Match imports with installed packages
    requirements = []
    for imp in code_imports:
        if imp.lower() in installed_packages:
            requirements.append(f"{imp}=={installed_packages[imp.lower()]}")
    
    # Write to file
    with open('requirements.txt', 'w') as f:
        f.write('\n'.join(sorted(requirements)))
    
    print(f"Generated requirements.txt with {len(requirements)} packages")

if __name__ == "__main__":
    generate_requirements()
