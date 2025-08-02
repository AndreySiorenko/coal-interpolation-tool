#!/usr/bin/env python3
"""
Temporary fix - disable database_connectors module to allow the app to start.
"""

from pathlib import Path

def disable_database_module():
    """Temporarily disable the problematic database module."""
    
    # Path to the __init__.py file
    init_path = Path(__file__).parent / "src" / "io" / "__init__.py"
    
    if not init_path.exists():
        print("❌ __init__.py не найден")
        return False
    
    # Read the file
    with open(init_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create backup
    backup_path = init_path.with_suffix('.py.backup')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Создан backup: {backup_path}")
    
    # Comment out the problematic import
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        if 'from .database_connectors import' in line:
            new_lines.append('# ' + line + '  # Temporarily disabled due to syntax error')
        else:
            new_lines.append(line)
    
    # Write the modified file
    new_content = '\n'.join(new_lines)
    with open(init_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Модуль database_connectors временно отключен")
    print("📝 Программа должна запуститься без этого модуля")
    return True

if __name__ == "__main__":
    disable_database_module()