from pathlib import Path
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


def index_project_knowledge():
    """Indexes source code, documentation, and research notes into ChromaDB."""
    # Find project root relative to this script
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent

    persist_directory = project_root / "vector_db"

    print(f"Starting to index project knowledge into {persist_directory}...")

    # Ensure the vector DB directory exists
    persist_directory.mkdir(parents=True, exist_ok=True)

    # 1. Load Documents
    documents = []

    # Load Python code files
    print("Loading Python code files...")
    src_dir = project_root / "src"
    if src_dir.exists():
        python_loader = DirectoryLoader(
            str(src_dir), glob="**/*.py", loader_cls=TextLoader, recursive=True
        )
        documents.extend(python_loader.load())

    # Load Markdown files
    print("Loading Markdown files...")
    markdown_loader = DirectoryLoader(
        str(project_root),
        glob="**/*.md",
        loader_cls=TextLoader,
        recursive=True,
        exclude=["**/node_modules/**", "**/.venv/**"],
    )
    documents.extend(markdown_loader.load())

    # Load RST files (Sphinx documentation)
    docs_source = project_root / "docs" / "source"
    if docs_source.exists():
        print("Loading reStructuredText files...")
        rst_loader = DirectoryLoader(
            str(docs_source), glob="**/*.rst", loader_cls=TextLoader, recursive=True
        )
        documents.extend(rst_loader.load())

    # Load JSON schema and config files
    print("Loading JSON schema and config files...")
    schema_dir = project_root / "schema"
    if schema_dir.exists():
        json_schema_loader = DirectoryLoader(
            str(schema_dir), glob="**/*.json", loader_cls=TextLoader, recursive=True
        )
        documents.extend(json_schema_loader.load())

    # Load research files
    research_dir = project_root / "research"
    if research_dir.exists():
        research_loader = DirectoryLoader(
            str(research_dir), glob="**/*.md", loader_cls=TextLoader, recursive=True
        )
        documents.extend(research_loader.load())

    # Load external literature
    literature_dir = project_root / "data" / "literature"
    if literature_dir.exists():
        print("Loading external literature files...")
        literature_loader = DirectoryLoader(
            str(literature_dir), glob="**/*.txt", loader_cls=TextLoader, recursive=True
        )
        documents.extend(literature_loader.load())

    print(f"Loaded {len(documents)} raw documents.")

    if not documents:
        print("No documents found to index.")
        return

    # 2. Split Documents
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    # 3. Generate Embeddings and Store in Vector Store
    print("Generating embeddings and storing in ChromaDB...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # ChromaDB has a batch size limit (around 5461)
    batch_size = 5000

    # Check if DB already exists
    if persist_directory.exists() and any(persist_directory.iterdir()):
        print("ChromaDB directory already exists. Attempting to load and update.")
        db = Chroma(
            persist_directory=str(persist_directory), embedding_function=embeddings
        )
        # Add documents in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            db.add_documents(batch)
    else:
        # Create first batch
        db = Chroma.from_documents(
            texts[:batch_size], embeddings, persist_directory=str(persist_directory)
        )
        # Add remaining in batches
        for i in range(batch_size, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            db.add_documents(batch)

    print("Project knowledge indexed successfully!")


def main():
    index_project_knowledge()


if __name__ == "__main__":
    main()
