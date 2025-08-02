#!/usr/bin/env python3
"""
Fix syntax error in database_connectors.py
"""

import ast
import sys
from pathlib import Path

def check_syntax(file_path):
    """Check syntax of a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to parse the file
        ast.parse(content)
        print(f"✅ {file_path} - синтаксис корректный")
        return True
    except SyntaxError as e:
        print(f"❌ {file_path} - синтаксическая ошибка:")
        print(f"   Строка {e.lineno}: {e.text.strip() if e.text else 'N/A'}")
        print(f"   Ошибка: {e.msg}")
        return False, e.lineno, e.text
    except Exception as e:
        print(f"❌ {file_path} - ошибка чтения: {e}")
        return False

def fix_database_connectors():
    """Fix the database_connectors.py file."""
    base_path = Path(__file__).parent
    file_path = base_path / "src" / "io" / "database_connectors.py"
    
    print("🔧 Проверка database_connectors.py...")
    
    result = check_syntax(file_path)
    if result is True:
        print("✅ Файл уже корректный!")
        return True
    
    is_ok, line_no, error_text = result
    if not is_ok:
        print(f"\n🔧 Попытка исправления ошибки в строке {line_no}...")
        
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Create backup
        backup_path = file_path.with_suffix('.py.backup')
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"✅ Создан backup: {backup_path}")
        
        # Try to fix common issues
        fixed = False
        if line_no <= len(lines):
            problematic_line = lines[line_no - 1]
            print(f"Проблемная строка {line_no}: {problematic_line.strip()}")
            
            # Common fixes
            new_line = problematic_line
            
            # Fix unclosed quotes
            if new_line.count('"') % 2 == 1:
                new_line = new_line.rstrip() + '"\n'
                fixed = True
                print("🔧 Исправлено: добавлена закрывающая кавычка")
            
            # Fix unclosed f-strings
            if 'f"' in new_line and new_line.count('"') % 2 == 1:
                new_line = new_line.rstrip() + '"\n'
                fixed = True
                print("🔧 Исправлено: f-string")
            
            if fixed:
                lines[line_no - 1] = new_line
                
                # Write the fixed file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                # Check again
                if check_syntax(file_path):
                    print("✅ Синтаксическая ошибка исправлена!")
                    return True
    
    print("❌ Не удалось автоматически исправить ошибку")
    return False

if __name__ == "__main__":
    success = fix_database_connectors()
    sys.exit(0 if success else 1)