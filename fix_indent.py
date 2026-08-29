# Fix indentation in demo_memory.py
import re

with open('demo_memory.py', 'rb') as f:
    content = f.read()

# Replace tabs with 4 spaces
content = content.replace(b'\t', b'    ')

# Fix specific known problematic lines
# The issue is that some lines have no indentation when they should
# Fix the specific problematic pattern: lines that should be indented but aren't

# Write back
with open('demo_memory.py', 'wb') as f:
    f.write(content)

print("Fixed tabs to spaces")

# Now let's also fix the specific indentation issues by reading line by line
with open('demo_memory.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed = []
for i, line in enumerate(lines):
    # Fix lines that start with 'print' or other keywords but have no indentation
    # when they should be inside a function (should have 4 spaces)
    if line.startswith('print(f"   Agent: {resp.content}")'):
        if not line.startswith('    '):
            line = '    ' + line
    elif line.startswith('print(f"   Agent: {resp.content}")'):
        if not line.startswith('    '):
            line = '    ' + line
    elif line.startswith('print("\\n[Waiting for background extraction to complete...]")'):
        if not line.startswith('    '):
            line = '    ' + line
    elif line.startswith('for _ in range(20):'):
        if not line.startswith('    '):
            line = '    ' + line
    elif line.startswith('if store.count_jobs(status="pending") == 0:'):
        if not line.startswith('        '):
            line = '        ' + line
    elif line.startswith('break'):
        if not line.startswith('            '):
            line = '            ' + line
    elif line.startswith('await asyncio.sleep(0.1)'):
        if not line.startswith('            '):
            line = '            ' + line
    elif line.startswith('# Recall Demonstration'):
        if not line.startswith('    '):
            line = '    ' + line
    elif line.startswith('print("\\nRecall Demonstration:'):
        if not line.startswith('    '):
            line = '    ' + line
    elif line.startswith('print("-" * 50)'):
        if not line.startswith('    '):
            line = '    ' + line
    elif line.startswith('resp = await wrapped.complete(['):
        if not line.startswith('    '):
            line = '    ' + line
    elif line.startswith('LLMMessage.user("What is my favorite color?")'):
        if not line.startswith('        '):
            line = '        ' + line
    elif line.startswith('], system="You are a helpful assistant.")'):
        if not line.startswith('    '):
            line = '    ' + line
    elif line.startswith('print(f"   Agent: {resp.content}")'):
        if not line.startswith('    '):
            line = '    ' + line
    elif line.startswith('print(f"   [System prompt included memory injection:]"):'):
        if not line.startswith('    '):
            line = '    ' + line
    elif line.startswith('sys_prompt = provider.systems[-1]'):
        if not line.startswith('    '):
            line = '    ' + line
    elif line.startswith('if sys_prompt and "Known context about this user:" in sys_prompt:'):
        if not line.startswith('    '):
            line = '    ' + line
    elif line.startswith('for line in sys_prompt.split("\\n"):'):
        if not line.startswith('        '):
            line = '        ' + line
    elif line.startswith('if line.strip().startswith("- ["):'):
        if not line.startswith('            '):
            line = '            ' + line
    elif line.startswith('print(f"      {line.strip()}"):'):
        if not line.startswith('            '):
            line = '            ' + line
    elif line.startswith('# Knowledge Graph Construction'):
        if not line.startswith('    '):
            line = '    ' + line
    elif line.startswith('print("\\nKnowledge Graph Construction")'):
        if not line.startswith('    '):
            line = '    ' + line
    elif line.startswith('print("-" * 50)'):
        if not line.startswith('    '):
            line = '    ' + line
    elif line.startswith('triple_store = TripleStore()'):
        if not line.startswith('    '):
            line = '    ' + line
    elif line.startswith('triple_store.add_triple(Triple('):
        if not line.startswith('    '):
            line = '    ' + line
    elif line.startswith('subject="user", predicate="prefers", object="blue",'):
        if not line.startswith('        '):
            line = '        ' + line
    elif line.startswith('confidence=0.9, source="conversation"'):
        if not line.startswith('        '):
            line = '        ' + line
    elif line.startswith('triple_store.add_triple(Triple('):
        if not line.startswith('    '):
            line = '    ' + line
    elif line.startswith('subject="user", predicate="uses_database", object="postgresql",'):
        if not line.startswith('        '):
            line = '        ' + line
    elif line.startswith('confidence=0.85, source="conversation"'):
        if not line.startswith('        '):
            line = '        ' + line
    elif line.startswith('triple_store.add_triple(Triple('):
        if not line.startswith('    '):
            line = '    ' + line
    elif line.startswith('subject="postgresql", predicate="is_a", object="database",'):
        if not line.startswith('        '):
            line = '        ' + line
    elif line.startswith('confidence=0.99, source="code_analysis"'):
        if not line.startswith('        '):
            line = '        ' + line
    elif line.startswith('print(f"   Triples: {triple_store.triple_count}")'):
        if not line.startswith('    '):
            line = '    ' + line
    elif line.startswith('print(f"   Nodes: {triple_store.node_count}")'):
        if not line.startswith('    '):
            line = '    ' + line
    elif line.startswith('expanded = triple_store.expand_query_with_graph("postgresql", max_hops=2)'):
        if not line.startswith('    '):
            line = '    ' + line
    elif line.startswith('print(f"   Expanded from \'postgresql\': {len(expanded)} related triples")'):
        if not line.startswith('    '):
            line = '    ' + line
    elif line.startswith('for t in expanded[:5]:'):
        if not line.startswith('    '):
            line = '    ' + line
    elif line.startswith('print(f"      {t.subject} -> {t.predicate} -> {t.object} (conf: {t.confidence:.2f})")'):
        if not line.startswith('        '):
            line = '        ' + line

    fixed.append(line)

with open('demo_memory.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed)

print("Fixed indentation issues")