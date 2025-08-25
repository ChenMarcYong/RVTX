import os, sys, time, shutil, re

if len(sys.argv) > 1 and (sys.argv[1] == "-h" or sys.argv[1] == "-help"):
    print("Usage: python watcher.py [<rvtx compilation folder> (optinnal, defaults to 'rVTX/out/build/x64-Release')] [refresh_time (optionnal, default 1)]")
    sys.exit(1)

root_folder = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.realpath(__file__) + "/../../../../out/build/x64-Release")
folders = [root_folder + "\\bin\\", root_folder + "\\lib\\"]

files_regex = re.compile('(.*pyd$)|(.*dll$)|(.*ptx$)|(.*frag$)|(.*geom$)|(.*vert$)')

watched_files = []
for folder in folders:
    for root, dirs, files in os.walk(folder):
        for file in files:
            if files_regex.match(file):
                watched_files.append({ "path": os.path.join(root, file), "filename" : os.path.join(root, file).replace(folder, "")  })

print("Watching:")
for watched_file in watched_files:
    print("  - " + watched_file["filename"])

refresh_time = 1 if len(sys.argv) < 3 else int(sys.argv[2])

print("Updates:")
while True:
    for watched_file in watched_files:
        try:
            if not os.path.exists(watched_file["filename"]):
                if os.path.dirname(watched_file["filename"]) != "":
                        os.makedirs(os.path.dirname(watched_file["filename"]), exist_ok=True)
                shutil.copy2(watched_file["path"], watched_file["filename"])
                print("  - [" + time.strftime("%H:%M:%S", time.gmtime()) + "] File '" + watched_file["filename"] + "' has been updated")
            elif os.stat(watched_file["filename"]).st_mtime < os.stat(watched_file["path"]).st_mtime:
                shutil.copy2(watched_file["path"], watched_file["filename"])
                print("  - [" + time.strftime("%H:%M:%S", time.gmtime()) + "] File '" + watched_file["filename"] + "' has been updated")
        except:
            pass

    time.sleep(refresh_time)
