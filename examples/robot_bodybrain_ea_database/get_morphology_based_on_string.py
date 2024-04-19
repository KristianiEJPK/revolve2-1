import concurrent.futures
from get_morphology_based_on_string_function import main_morph
import os


def main(files, path, cpu_count):
    # ---- Get morphological measures
    print(f"Starting with {cpu_count} workers")

    with concurrent.futures.ProcessPoolExecutor(max_workers = cpu_count
                    ) as executor:
        futures_and_key = {executor.submit(main_morph, filepath, file_key, path + f"\\morphological_measures_{file_key}.csv"):file_key for file_key, filepath in files.items()}


        
           
        # # ---- Save data
        # for future in concurrent.futures.as_completed(futures_and_key):
        #     file_key = futures_and_key[future]
        #     df = future.result()

            

if __name__ == "__main__":
    # ---- Set path
    path = "C:\\Users\\niels\\OneDrive\\Documenten\\GitHub\\revolve2"#"D:\\AI\\Random\\Random"
    cpu_count = os.cpu_count()

    # ---- Get all files
    files = {}
    for file in os.listdir(path):
        if file.endswith(".json") and ("_counts" not in file) and ("morphological_measures" not in file):
            parts = int(file.split("_")[1].split(".")[0])
            files[parts] = os.path.join(path, file)
            #assert not os.path.exists(path + f"\\morphological_measures_{parts}.csv"), "File already exists!"


    # ---- Run main
    main(files, path, cpu_count = cpu_count)
