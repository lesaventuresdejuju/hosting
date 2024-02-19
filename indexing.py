from azure_utils import AzureFactory
from document_loaders import UnstructuredDocumentLoader
from llama_index import (
    ServiceContext, 
    StorageContext,
    VectorStoreIndex,
    Document
)
from llama_index.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
    UnstructuredElementNodeParser
)
from llama_index.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor
)


DIRECTORY = "/Users/jwang/Desktop/File Categorization Example/nl1/"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128

documents = UnstructuredDocumentLoader().load_directory(DIRECTORY)
print(f"Loaded {len(documents)} documents")

vector_store = AzureFactory.getVectorStore()
llm = AzureFactory.getLLM()
embed_model = AzureFactory.getEmbedding()
index_client =  AzureFactory.getSearchIndexClient()
search_client = AzureFactory.getSearchClient()

# Metadata extractors
text_splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
title_extractor = TitleExtractor(nodes=1, llm=llm)
qa_extractor = QuestionsAnsweredExtractor(questions=3, llm=llm)
summary_extractor = SummaryExtractor(summaries=["prev", "self"], llm=llm)
keyword_extractor = KeywordExtractor(keywords=10, llm=llm)

# Service context with metadata extractors
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    transformations=[
        text_splitter,
        title_extractor,
        qa_extractor,
        summary_extractor,
        keyword_extractor,
    ]
)
# service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


# node_parser = UnstructuredElementNodeParser()
# raw_nodes = node_parser.get_nodes_from_documents(documents=documents)
# base_nodes, node_mappings = node_parser.get_base_nodes_and_mappings(raw_nodes)

index = VectorStoreIndex.from_documents(
    documents=documents,
    service_context=service_context,
    storage_context=storage_context,
    show_progress=True
)

query_engine = index.as_query_engine()
response = query_engine.query("what is the guide about?")
print(response)