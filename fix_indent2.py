# Comprehensive fix for demo_memory.py
import re

with open('demo_memory.py', 'rb') as f:
    content = f.read()

# Replace all \r\n with \n
content = content.replace(b'\r\n', b'\n')

# Replace all tabs with 4 spaces
content = content.replace(b'\t', b'    ')

with open('demo_memory.py', 'wb') as f:
    f.write(content)

print("Normalized line endings and tabs")

# Now fix specific indentation issues line by line
with open('demo_memory.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed = []
for line in lines:
    # Ensure all lines that should be indented have proper 4-space indentation
    # Skip empty lines
    if line.strip() == '':
        fixed.append('\n')
        continue
    
    # Fix lines that should have 4-space indent but don't
    # Lines that are at module level (def, class, import, async) should have no indent
    # Lines inside functions should have 4 spaces
    # Lines inside nested blocks should have 8 spaces, etc.
    
    # Detect if line should be at module level (no indent)
    stripped = line.lstrip()
    if stripped.startswith(('def ', 'class ', 'import ', 'from ', 'async def ', '@')):
        # Module level - no indent
        fixed.append(stripped)
    elif line.startswith('    ') or line.startswith('        '):
        # Already indented - keep as is (but normalize)
        fixed.append(line.expandtabs(4))
    elif stripped and not line.startswith(' '):
        # Line has content but no indent - likely should be indented
        # This is a heuristic - if it's a statement inside a function
        if any(keyword in stripped for keyword in ['print', 'await', 'if ', 'for ', 'while ', 'try:', 'except:', 'with ', 'return', 'raise', 'store.', 'layer.', 'worker.', 'resp =', 'user_msg', 'resp ', 'sys_prompt', 'if sys_prompt', 'for line', 'if line.strip']):
            fixed.append('    ' + stripped)
        else:
            fixed.append(stripped)
    else:
        # Normalize tabs
        fixed.append(line.expandtabs(4))

with open('demo_memory.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed)

print("Fixed")