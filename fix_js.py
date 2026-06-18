import sys
import re

with open('templates/admin_live_dashboard.html', 'r', encoding='utf-8') as f:
    text = f.read()

pattern = re.compile(r'function updateStudentsGrid\(\) \{[\s\S]*?grid\.innerHTML = \'\';', re.MULTILINE)
match = pattern.search(text)
if match:
    old_func = match.group(0)
    print('FOUND EXACT BLOCK:')
    print(repr(old_func))
    
    new_func = old_func.replace(
        "document.getElementById('loading-message').style.display = 'block';\n                document.getElementById('loading-message').innerHTML =\n                    '<p>🔄 Waiting for students to start exams...</p>';",
        "if (loadingMsg) {\n                    loadingMsg.style.display = 'block';\n                    loadingMsg.innerHTML = '<p>🔄 Waiting for students to start exams...</p>';\n                } else {\n                    grid.innerHTML = '<div class=\"loading\" id=\"loading-message\"><p>🔄 Waiting for students to start exams...</p></div>';\n                }"
    )
    # also add loadingMsg definition
    if 'const loadingMsg = document.getElementById(' not in new_func:
        new_func = new_func.replace(
            "const students = Object.values(studentsData);",
            "const students = Object.values(studentsData);\n            const loadingMsg = document.getElementById('loading-message');"
        )
    
    text = text.replace(old_func, new_func)
    
    with open('templates/admin_live_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(text)
    print('PATCH APPLIED!')
else:
    print('COULD NOT FIND FUNCTION!')
