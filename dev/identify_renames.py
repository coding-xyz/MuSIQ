import subprocess
import re

def main():
    try:
        diff = subprocess.check_output(['git', 'diff', '56be4a5^', '56be4a5', '--', 'src/musiq']).decode('utf-8')
    except subprocess.CalledProcessError as e:
        print(f"Error running git diff: {e}")
        return

    lines = diff.splitlines()
    renames = []
    current_minus = None
    
    for line in lines:
        if line.startswith('- class '):
            current_minus = line[2:].strip()
        elif line.startswith('+ class '):
            if current_minus:
                renames.append((current_minus, line[2:].strip()))
                current_minus = None
            else:
                # New class added, not a rename
                pass
        elif line.startswith('+') or line.startswith('-'):
            # If we hit another change line that isn't a '+ class', 
            # we reset the current_minus if it was set.
            if not line.startswith('+ class '):
                current_minus = None
        elif not line.startswith(' '):
            # We hit a hunk header or file header
            current_minus = None

    if not renames:
        print("No class renames found using simple matching. Trying a more liberal approach...")
        # Fallback: just list all class removals and additions
        removals = [l[2:].strip() for l in lines if l.startswith('- class ')]
        additions = [l[2:].strip() for l in lines if l.startswith('+ class ')]
        print("\nRemoved classes:")
        for r in removals: print(f"  - {r}")
        print("\nAdded classes:")
        for a in additions: print(f"  + {a}")
    else:
        print("Detected Class Renames:")
        for old, new in renames:
            print(f"{old} -> {new}")

if __name__ == "__main__":
    main()