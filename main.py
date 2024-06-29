from sentence_transformers import SentenceTransformer, util
import chromadb
import random
from chromadb.utils import embedding_functions
import networkx as nx

# Constants and configuration
CHROMA_DATA_PATH = "chroma_data/"
COLLECTION_NAME = "TaskEmbeddings"
# Initialize the sentence transformer embedding function
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="multi-qa-mpnet-base-dot-v1"
)

# Set up ChromaDB client and collection
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection = client.get_or_create_collection(
    name="TaskEmbeddings",
    metadata={"hnsw:space": "cosine"},
    embedding_function=sentence_transformer_ef,
)


def save_embedding_to_chromadb(task):
    """
    Save a task's embedding to ChromaDB.

    :param task: Dictionary containing task information
    """
    collection.add(
        documents=task["taskdescription"],
        metadatas={"EmployeeId": task["empid"]},
        ids=task["taskid"],
    )


def textrank_truncate(text, max_tokens=256):
    """
    Truncate text using TextRank algorithm to preserve most important information.

    :param text: Input text to truncate
    :param max_tokens: Maximum number of tokens in the truncated text
    :return: Truncated text
    """
    # Split text into sentences
    sentences = text.split(".")

    # Initialize sentence transformer model
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Generate embeddings for each sentence
    embeddings = embedder.encode(sentences, convert_to_tensor=True)

    # Calculate cosine similarity between all pairs of sentences
    similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings).numpy()

    # Create a graph from the similarity matrix
    graph = nx.from_numpy_array(similarity_matrix)

    # Apply PageRank algorithm to rank sentences
    scores = nx.pagerank(graph)

    # Sort sentences based on their PageRank scores
    ranked_sentences = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)), reverse=True
    )

    # Construct truncated text by adding sentences until token limit is reached
    truncated_text = ""
    token_count = 0
    for _, sentence in ranked_sentences:
        sentence_tokens = len(sentence.split())
        if token_count + sentence_tokens <= max_tokens:
            truncated_text += sentence + "."
            token_count += sentence_tokens
        else:
            break

    return truncated_text.strip()


# Sample task descriptions
taskdescs = [
    "Schedule and organize a project kickoff meeting to introduce team members and discuss project goals.",
    "Conduct interviews with key stakeholders to gather requirements and understand their expectations.",
    "Develop a detailed project timeline outlining key milestones and deadlines.",
    "Identify potential risks and uncertainties associated with the project and develop a risk mitigation plan.",
    "Clearly define the scope of the project, including deliverables, features, and exclusions.",
    "Allocate resources effectively, considering team members' skills, availability, and project requirements.",
    "Evaluate and choose the appropriate technology stack for the project, considering scalability and compatibility.",
    "Create a prototype or wireframe to visualize the project's user interface and functionality.",
    "Set up the project's version control system, development environment, and coding standards.",
    "Design the database schema and establish data relationships for efficient data management.",
    "Initiate frontend development, implementing the user interface and ensuring a responsive design.",
    "Begin backend development, focusing on server-side logic, database integration, and API development.",
    "Plan and conduct UAT sessions with stakeholders to validate that the system meets their requirements.",
    "Implement a comprehensive QA plan, including testing protocols, to ensure the software's reliability and functionality.",
    "Generate project documentation, including user manuals, technical documentation, and API documentation.",
    "Develop a deployment plan outlining the steps to release the project to production.",
    "Conduct training sessions for end-users and stakeholders on how to use the new system effectively.",
    "Address and resolve any bugs or issues identified during testing, and optimize system performance.",
    "Hold a project review meeting to assess the overall success of the project, gather feedback, and discuss lessons learned.",
]

# Process and store tasks
tasks = []
for i, desc in enumerate(taskdescs):
    # Create task dictionary with truncated description
    task = {
        "taskdescription": textrank_truncate(desc),
        "empid": f"emp{random.randrange(1,19)}",  # Assign random employee ID
        "taskid": f"task{i+1}",  # Assign sequential task ID
    }
    tasks.append(task)
    # Save task embedding to ChromaDB
    save_embedding_to_chromadb(task)

# Define a new task to be assigned
tasktbd_original = "Assign specific tasks to team members based on their expertise and the project requirements."

# Truncate the task to be done
tasktbd_truncated = textrank_truncate(tasktbd_original)

# Query ChromaDB for similar tasks using the truncated version
results = collection.query(query_texts=tasktbd_truncated, n_results=5)
print(results)
