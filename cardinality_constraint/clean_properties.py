def remove_unwanted_properties(
    input_file="multivalue_properties.txt",
    output_file="filtered_multivalue_properties.txt"
):
    # Substrings to filter out (case-insensitive)
    blacklist = ["id", "code", "number", "key", "url", "iswc", "index", "username", "doi", "rate"]

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    filtered_lines = [
        line for line in lines
        if not any(bad_word in line.lower() for bad_word in blacklist)
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(filtered_lines)

    print(f"Filtered list saved to {output_file}. {len(filtered_lines)} properties remain.")

if __name__ == "__main__":
    remove_unwanted_properties()


