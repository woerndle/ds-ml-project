import os

# List of common files and directories to ignore
IGNORE_LIST = {'.git', '.gitignore', '__pycache__', '.env', '.DS_Store', 'node_modules'}

def should_ignore(item):
    """Check if an item should be ignored based on the ignore list."""
    return item in IGNORE_LIST

def generate_context(directory, indent_level=0):
    """Recursively generates a string representation of the directory structure and its contents."""
    context = ""
    indent = '  ' * indent_level  # Indentation for readability
    for item in os.listdir(directory):
        path = os.path.join(directory, item)
        if should_ignore(item):
            continue  # Skip ignored items
        
        if os.path.isdir(path):
            # If it's a directory, append its name and recurse into it
            context += f"{indent}Directory: {item}\n"
            context += generate_context(path, indent_level + 1)  # Recur for the directory
        elif os.path.isfile(path):
            # If it's a file, append its name and its content
            context += f"{indent}File: {item}\n"
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    # Limit content length and format it correctly
                    limited_content = content[:1000].replace('\n', '\n' + indent + '  ')
                    context += f"{indent}  Content:\n{indent}  {limited_content}\n"  # Include formatted content
            except Exception as e:
                context += f"{indent}  Content: [Could not read file: {e}]\n"
    return context

def write_context_file(output_file):
    """Generates the context and writes it to a file."""
    context = generate_context(os.getcwd())  # Start from the current working directory
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(context)
    print(f"Context has been written to {output_file}")

if __name__ == "__main__":
    output_filename = "context.txt"
    write_context_file(output_filename)
