import os
from pathlib import Path
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def _get_paths():
    # Find project root relative to this script
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    return {
        "vector_db": project_root / "vector_db",
        "templates": project_root / "prompts" / "templates",
        "root": project_root,
    }


def _load_rag_prompt_template(template_name: str) -> ChatPromptTemplate:
    """
    Loads a RAG prompt template from the templates directory.
    """
    paths = _get_paths()
    template_path = paths["templates"] / f"{template_name}.md"
    if not template_path.exists():
        # Fallback
        template_path = Path("chrysalis/prompts/templates") / f"{template_name}.md"
        if not template_path.exists():
            raise FileNotFoundError(
                f"RAG prompt template '{template_name}.md' not found."
            )

    with open(template_path, "r", encoding="utf-8") as f:
        template_content = f.read()

    return ChatPromptTemplate.from_template(template_content)


def _load_rag_pipeline(llm_model_name: str, template_name: str) -> RunnablePassthrough:
    """
    Loads the ChromaDB, Ollama Embeddings, Ollama LLM, and sets up the RAG chain.
    """
    paths = _get_paths()
    vector_db_dir = paths["vector_db"]

    if not vector_db_dir.exists():
        # Fallback
        vector_db_dir = Path("chrysalis/vector_db")
        if not vector_db_dir.exists():
            raise FileNotFoundError(
                f"Vector database not found at {vector_db_dir}. Please run index_project.py first."
            )

    print(
        f"Initializing Ollama Embeddings (nomic-embed-text) and loading ChromaDB from {vector_db_dir}..."
    )
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=str(vector_db_dir), embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever()

    print(f"Initializing Ollama LLM ({llm_model_name})...")
    llm = OllamaLLM(model=llm_model_name)

    # Define the RAG prompt template
    prompt = _load_rag_prompt_template(template_name)

    # Create the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def chat_with_project(
    query: str,
    llm_model_name: str = "llama3.3:70b-instruct-q4_K_M",
    template_name: str = "rag_default_template",
) -> str:
    """
    Answers a question about the Chrysalis project using RAG with a local Ollama LLM.
    """
    rag_pipeline = _load_rag_pipeline(llm_model_name, template_name)
    print(f"Generating response using {llm_model_name}...")
    response = rag_pipeline.invoke(query)
    return response


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Chat with the Chrysalis project knowledge base using RAG."
    )
    parser.add_argument(
        "query", type=str, help="The question to ask about the Chrysalis project."
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="llama3.3:70b-instruct-q4_K_M",
        help="The Ollama LLM model to use for generation.",
    )
    parser.add_argument(
        "--ollama_base_url",
        type=str,
        default="http://localhost:11434",
        help="Base URL for the Ollama server.",
    )
    parser.add_argument(
        "--template_name",
        type=str,
        default="rag_default_template",
        help="Name of the RAG prompt template file.",
    )

    args = parser.parse_args()

    # Set OLLAMA_BASE_URL
    os.environ["OLLAMA_BASE_URL"] = args.ollama_base_url

    print("--- Chrysalis Project RAG Chat ---")
    try:
        response = chat_with_project(args.query, args.llm_model, args.template_name)
        print("\n--- Answer ---")
        print(response)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(
            "Please ensure you have indexed the project knowledge by running `index-rag` first."
        )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
