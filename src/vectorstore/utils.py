def show_results(store, query: str, top_k: int = 4) -> None:
    results = store.hybrid_search(query, top_k=top_k)
    print(f"Query: {query!r}\n")
    print(f"{'#':<3} {'Score':>6}  {'Kind':<12} {'Symbol / Title':<35} Citation")
    print("─" * 110)
    for i, (payload, score) in enumerate(results, 1):
        label = payload.get("symbol") or payload.get("page_title", "—")
        print(
            f"{i:<3} {score:>6.4f}  "
            f"{payload.get('kind', '?'):<12} "
            f"{label[:45]:<35} "
            f"{payload.get('citation_url', '')}"
        )
    print()
