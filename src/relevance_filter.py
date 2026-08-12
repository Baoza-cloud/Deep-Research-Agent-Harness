def filter_results(
    retrieved_results,
    max_distance=1.0,
    max_chunks=3
):

    filtered_results = []

    for result in retrieved_results:

        if result["distance"] <= max_distance:

            filtered_results.append(result)

        if len(filtered_results) >= max_chunks:
            break

    return filtered_results