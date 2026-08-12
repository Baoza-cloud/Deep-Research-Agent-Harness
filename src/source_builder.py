def build_sources(filtered_results):

    sources = []

    seen = set()

    for result in filtered_results:

        source = result["source"]

        if source not in seen:

            sources.append(source)
            seen.add(source)

    return sources
