from tools.retriever import RetrieverTool

if __name__ == "__main__":
    retriever = RetrieverTool()
    query = "Where is the parade?"
    results = retriever.query(query)

    print("\n📄 Top Retrieved Passages:")
    for i, passage in enumerate(results, 1):
        print(f"\n[{i}] {passage[:300]}...\n")