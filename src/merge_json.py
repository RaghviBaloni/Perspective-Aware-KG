import json

def merge_annotation_files(success_file, failed_file, output_file):
    # Load successful annotations
    with open(success_file, "r") as f:
        success_data = json.load(f)
        success = success_data.get("annotations", {})

    # Load failed annotations (list of objects)
    with open(failed_file, "r") as f:
        failed_list = json.load(f)

    failed = {}

    # Convert each failed entry into an empty annotation
    for item in failed_list:
        tweet_id = str(item.get("tweet_id"))
        if tweet_id:
            failed[tweet_id] = {
                "annotation": {
                    tweet_id: []   # empty list = no perspectives
                },
                "raw_response": item.get("raw_response", ""),
                "error": item.get("error", "")
            }

    # Merge successful + failed
    merged = {**success, **failed}

    # Save merged file
    with open(output_file, "w") as f:
        json.dump({"annotations": merged}, f, indent=2)

    print(f"Merged {len(success)} success + {len(failed)} failed → {len(merged)} total")

if __name__ == "__main__":
    merge_annotation_files(
        "/home/baloni/Perspective-Aware-KG/data/annotations/pass1_llama-3.1-70b-awq_full_20260119_123634.json",
        "/home/baloni/Perspective-Aware-KG/data/annotations/pass1_llama-3.1-70b-awq_full_20260119_123634_failed.json",
        "/home/baloni/Perspective-Aware-KG/data/processed/merged_annotations_1.json"
    )
