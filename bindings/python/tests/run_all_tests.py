import glob, os, sys, traceback
from colorama import Fore, Style

excluded = ["run_all_tests.py", "watcher.py"]

os.chdir(os.path.dirname(os.path.realpath(__file__)))
count = len(glob.glob("*.py")) - len(excluded)
current, success = 1, 0
for script in sorted(glob.glob("*.py")):
    if script in excluded:
        continue
    with open(script) as f:
       contents = f.read()

    print()
    print(Fore.BLUE + f"({current}|{count}) Running '" + script + "' tests...")
    
    try:
        exec(contents)
        success += 1
        print(Fore.GREEN + f"  → '{script}' test passed!")
    except Exception as e:
        _, _, tb = sys.exc_info()
        _, line, _, _ = traceback.extract_tb(tb)[-1]
        line_text = contents.split('\n')[line - 1]

        print(Fore.RED + f"  Error line {line} in '{script}':")
        print(Fore.RED + f"   ↳ '{line_text}' failed!")

    current += 1

color = Fore.GREEN if success == count else Fore.RED
msg = "[SUCCESS]" if success == count else "[FAILURE]"

print()
print(color + f"{msg} > Passed {success} out of {count} test." + Style.RESET_ALL)
print()