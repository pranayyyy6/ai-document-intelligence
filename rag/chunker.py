from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(pages: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(pages)
    print(f"✅ Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks

def preview_chunks(chunks: list, n: int = 3):
    print(f"\n--- Previewing first {n} chunks ---")
    for i, chunk in enumerate(chunks[:n]):
        print(f"\nChunk {i+1} (Page {chunk.metadata.get('page', '?')+1}):")
        print(f"Content: {chunk.page_content[:150]}...")