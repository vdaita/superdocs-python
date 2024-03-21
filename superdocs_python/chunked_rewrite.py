from retriever import CodebaseRetriever

def chunked_rewrite(filepath, file, goal, model):
    cbretriever = CodebaseRetriever("")
    chunks = cbretriever.get_chunks_for_file(file, filepath)
    reconstructed_file = "\n".join(f"[Chunk {index}]\n```\n{chunk}\n```\n" for index, chunk in enumerate(chunks))
    
    print(f"File: {filepath}")
    print(reconstructed_file)

if __name__ == "__main__":
    filepath = "/Users/vijaydaita/Files/uiuc/rxassist/rxassist/src/app/main/page.tsx"
    contents = open(filepath, "r").read()
    chunked_rewrite(filepath, contents, "", "")